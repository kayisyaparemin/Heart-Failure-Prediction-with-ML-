package com.navisun.fueltracker.service

import android.Manifest
import android.app.*
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import androidx.core.app.ActivityCompat
import androidx.core.app.NotificationCompat
import androidx.localbroadcastmanager.content.LocalBroadcastManager
import com.navisun.fueltracker.MainActivity
import com.navisun.fueltracker.R
import com.navisun.fueltracker.data.FuelDatabase
import com.navisun.fueltracker.data.TripEntry
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Locale
import kotlin.math.asin
import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

// Detection thresholds
private const val START_SPEED_KMH = 5.0f
private const val STOP_SPEED_KMH = 3.0f
private const val CONFIRM_START_MS = 8_000L    // 8 seconds
private const val CONFIRM_STOP_MS = 180_000L   // 3 minutes
private const val MIN_TRIP_DISTANCE_KM = 0.3

// Checkpoint constants
private const val PREFS_CHECKPOINT = "trip_checkpoint"
private const val KEY_HAS_CHECKPOINT = "has_checkpoint"
private const val KEY_CP_START_TIME = "cp_start_time"
private const val KEY_CP_START_LAT = "cp_start_lat"
private const val KEY_CP_START_LON = "cp_start_lon"
private const val KEY_CP_ROUTE = "cp_route"
private const val KEY_CP_MAX_SPEED = "cp_max_speed"
private const val KEY_CP_DISTANCE = "cp_distance"
private const val KEY_CP_FUEL_TYPE = "cp_fuel_type"
private const val KEY_CP_SEGMENTS = "cp_segments"
private const val CHECKPOINT_INTERVAL_MS = 30_000L  // every 30 seconds

private data class SegmentStart(val fuelType: String, val startRouteIndex: Int, val startTimeMs: Long)

class TripTrackingService : Service() {

    companion object {
        const val ACTION_START = "com.navisun.fueltracker.START_TRACKING"
        const val ACTION_STOP = "com.navisun.fueltracker.STOP_TRACKING"
        const val ACTION_SIMULATE_TRIP = "com.navisun.fueltracker.SIMULATE_TRIP"
        const val ACTION_TRIP_STATE_UPDATE = "TRIP_STATE_UPDATE"
        const val ACTION_FUEL_TYPE_CHANGED = "com.navisun.fueltracker.FUEL_TYPE_CHANGED"
        const val EXTRA_STATE = "state"
        const val EXTRA_SPEED_KMH = "speed_kmh"
        const val EXTRA_DISTANCE_KM = "distance_km"
        const val EXTRA_FUEL_TYPE = "fuel_type"

        const val ACTION_SPEED_UPDATE = "TRIP_STATE_UPDATE"
        private const val NOTIFICATION_ID = 1001
        private const val CHANNEL_ID = "trip_channel"
    }

    private enum class TripState {
        IDLE, CONFIRMING_START, RECORDING, CONFIRMING_STOP
    }

    private data class RoutePoint(
        val lat: Double,
        val lon: Double,
        val time: Long,
        val speedKmh: Float
    )

    private lateinit var locationManager: LocationManager

    private var state = TripState.IDLE
    private var movingStartTime = 0L
    private var stoppedStartTime = 0L

    private var startLat = 0.0
    private var startLon = 0.0
    private var startTime = 0L
    private var routePoints = mutableListOf<RoutePoint>()
    private var maxSpeed = 0f
    private var activeFuelType = "LPG"

    private var segmentStarts = mutableListOf<SegmentStart>()
    private var isSimulationMode = false

    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    private val checkpointHandler = Handler(Looper.getMainLooper())
    private val checkpointRunnable = object : Runnable {
        override fun run() {
            if (state == TripState.RECORDING || state == TripState.CONFIRMING_STOP) {
                saveCheckpoint()
            }
            checkpointHandler.postDelayed(this, CHECKPOINT_INTERVAL_MS)
        }
    }

    private val fuelTypeReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            val newType = intent?.getStringExtra(EXTRA_FUEL_TYPE) ?: return
            if (newType == activeFuelType) return
            activeFuelType = newType
            if (state == TripState.RECORDING || state == TripState.CONFIRMING_STOP) {
                segmentStarts.add(SegmentStart(newType, routePoints.size, System.currentTimeMillis()))
            }
        }
    }

    private val locationListener = object : LocationListener {
        override fun onLocationChanged(location: Location) {
            handleLocation(location)
        }

        @Deprecated("Deprecated in Java")
        override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) {}
        override fun onProviderEnabled(provider: String) {}
        override fun onProviderDisabled(provider: String) {}
    }

    override fun onCreate() {
        super.onCreate()
        locationManager = getSystemService(LOCATION_SERVICE) as LocationManager
        createNotificationChannel()
        LocalBroadcastManager.getInstance(this).registerReceiver(
            fuelTypeReceiver, IntentFilter(ACTION_FUEL_TYPE_CHANGED)
        )
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        startForeground(NOTIFICATION_ID, buildNotification())
        recoverCheckpointIfExists()
        when (intent?.action) {
            ACTION_START -> beginTracking()
            ACTION_STOP -> endTracking()
            ACTION_SIMULATE_TRIP -> startSimulation()
        }
        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        super.onDestroy()
        serviceScope.cancel()
        checkpointHandler.removeCallbacks(checkpointRunnable)
        if (state == TripState.RECORDING && routePoints.isNotEmpty()) {
            saveCheckpoint()
        }
        LocalBroadcastManager.getInstance(this).unregisterReceiver(fuelTypeReceiver)
        try {
            locationManager.removeUpdates(locationListener)
        } catch (e: Exception) {
            // ignore
        }
    }

    override fun onTaskRemoved(rootIntent: Intent?) {
        super.onTaskRemoved(rootIntent)
        val restartIntent = Intent(applicationContext, TripTrackingService::class.java).apply {
            action = ACTION_START
        }
        val pendingIntent = PendingIntent.getService(
            applicationContext, 1, restartIntent,
            PendingIntent.FLAG_ONE_SHOT or PendingIntent.FLAG_IMMUTABLE
        )
        val alarmManager = getSystemService(ALARM_SERVICE) as AlarmManager
        val triggerAt = android.os.SystemClock.elapsedRealtime() + 1000
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            alarmManager.setExactAndAllowWhileIdle(AlarmManager.ELAPSED_REALTIME, triggerAt, pendingIntent)
        } else {
            alarmManager.setExact(AlarmManager.ELAPSED_REALTIME, triggerAt, pendingIntent)
        }
    }

    private fun beginTracking() {
        state = TripState.IDLE
        requestLocationUpdates(slow = true)
        checkpointHandler.removeCallbacks(checkpointRunnable)
        checkpointHandler.postDelayed(checkpointRunnable, CHECKPOINT_INTERVAL_MS)
    }

    private fun startSimulation() {
        if (state == TripState.RECORDING || state == TripState.CONFIRMING_STOP) return
        isSimulationMode = true
        state = TripState.IDLE

        // İstanbul Kadıköy → Bağdat Cd. güzergahı (~8.5km, ~12dk)
        // Her nokta arası 2.5 saniye → toplam ~75 sn hareket + 10 sn durma = ~85 sn
        data class SimPoint(val lat: Double, val lon: Double, val speedKmh: Float)
        val route = listOf(
            SimPoint(40.9900, 29.0280,  0f),
            SimPoint(40.9908, 29.0255, 28f),
            SimPoint(40.9918, 29.0220, 46f),
            SimPoint(40.9930, 29.0175, 62f),
            SimPoint(40.9944, 29.0120, 74f),
            SimPoint(40.9958, 29.0065, 81f),
            SimPoint(40.9972, 29.0008, 79f),
            SimPoint(40.9986, 28.9950, 76f),
            SimPoint(41.0000, 28.9892, 82f),
            SimPoint(41.0014, 28.9834, 88f),
            SimPoint(41.0028, 28.9776, 84f),
            SimPoint(41.0040, 28.9720, 77f),
            SimPoint(41.0052, 28.9664, 71f),
            SimPoint(41.0062, 28.9612, 65f),
            SimPoint(41.0070, 28.9572, 54f),
            SimPoint(41.0076, 28.9540, 42f),
            SimPoint(41.0080, 28.9518, 30f),
            SimPoint(41.0082, 28.9505, 14f),
            SimPoint(41.0083, 28.9500,  4f),
            SimPoint(41.0083, 28.9499,  1f),
            SimPoint(41.0083, 28.9499,  0f),
            SimPoint(41.0083, 28.9499,  0f),
            SimPoint(41.0083, 28.9499,  0f),
            SimPoint(41.0083, 28.9499,  0f)
        )

        serviceScope.launch {
            for (pt in route) {
                val loc = Location("simulation").apply {
                    latitude  = pt.lat
                    longitude = pt.lon
                    speed     = pt.speedKmh / 3.6f
                    time      = System.currentTimeMillis()
                    accuracy  = 8f
                }
                withContext(Dispatchers.Main) { handleLocation(loc) }
                delay(2500)
            }
        }
    }

    private fun endTracking() {
        checkpointHandler.removeCallbacks(checkpointRunnable)
        try {
            locationManager.removeUpdates(locationListener)
        } catch (e: Exception) {
            // ignore
        }
        stopForeground(true)
        stopSelf()
    }

    private fun handleLocation(location: Location) {
        val speedKmh = location.speed * 3.6f
        val now = System.currentTimeMillis()

        when (state) {
            TripState.IDLE -> {
                if (speedKmh > START_SPEED_KMH) {
                    movingStartTime = now
                    state = TripState.CONFIRMING_START
                    requestLocationUpdates(slow = false)
                }
            }

            TripState.CONFIRMING_START -> {
                val confirmMs = if (isSimulationMode) 2_000L else CONFIRM_START_MS
                if (speedKmh > START_SPEED_KMH) {
                    if (now - movingStartTime >= confirmMs) {
                        startTrip(location)
                        state = TripState.RECORDING
                    }
                } else {
                    state = TripState.IDLE
                    requestLocationUpdates(slow = true)
                }
            }

            TripState.RECORDING -> {
                routePoints.add(RoutePoint(location.latitude, location.longitude, now, speedKmh))
                if (speedKmh > maxSpeed) maxSpeed = speedKmh
                broadcastStateUpdate(state, speedKmh)
                updateNotification()
                if (speedKmh < STOP_SPEED_KMH) {
                    stoppedStartTime = now
                    state = TripState.CONFIRMING_STOP
                }
            }

            TripState.CONFIRMING_STOP -> {
                val stopMs = if (isSimulationMode) 8_000L else CONFIRM_STOP_MS
                if (speedKmh >= STOP_SPEED_KMH) {
                    state = TripState.RECORDING
                    routePoints.add(RoutePoint(location.latitude, location.longitude, now, speedKmh))
                    if (speedKmh > maxSpeed) maxSpeed = speedKmh
                } else {
                    if (now - stoppedStartTime >= stopMs) {
                        saveTrip(location)
                        isSimulationMode = false
                        state = TripState.IDLE
                        requestLocationUpdates(slow = true)
                    }
                }
                broadcastStateUpdate(state, speedKmh)
                updateNotification()
            }
        }

        if (state == TripState.IDLE || state == TripState.CONFIRMING_START) {
            broadcastStateUpdate(state, speedKmh)
            updateNotification()
        }
    }

    private fun startTrip(location: Location) {
        clearCheckpoint()
        startLat = location.latitude
        startLon = location.longitude
        startTime = System.currentTimeMillis()
        routePoints = mutableListOf()
        maxSpeed = 0f
        activeFuelType = getSharedPreferences("navisun_prefs", Context.MODE_PRIVATE)
            .getString("active_fuel_type", "LPG") ?: "LPG"
        segmentStarts = mutableListOf(SegmentStart(activeFuelType, 0, startTime))
    }

    private fun saveTrip(location: Location) {
        if (routePoints.size < 2) return

        var distance = 0.0
        for (i in 1 until routePoints.size) {
            distance += haversineKm(
                routePoints[i - 1].lat, routePoints[i - 1].lon,
                routePoints[i].lat, routePoints[i].lon
            )
        }
        if (distance < MIN_TRIP_DISTANCE_KM) return

        val endTime = System.currentTimeMillis()
        val durationMin = ((endTime - startTime) / 60000).toInt()
        val durationHours = (endTime - startTime) / 3_600_000.0
        val avgSpeed = if (durationHours > 0) (distance / durationHours).toFloat() else 0f

        val fuelType = getSharedPreferences("navisun_prefs", Context.MODE_PRIVATE)
            .getString("active_fuel_type", "LPG") ?: "LPG"

        val routeJson = serializeRoute(routePoints)
        val segJson = buildFinalSegmentsJson(endTime)

        val trip = TripEntry(
            startTime = startTime,
            endTime = endTime,
            startLat = startLat,
            startLon = startLon,
            endLat = location.latitude,
            endLon = location.longitude,
            distanceKm = distance,
            avgSpeedKmh = avgSpeed.toDouble(),
            maxSpeedKmh = maxSpeed.toDouble(),
            durationMinutes = durationMin,
            routePointsJson = routeJson,
            fuelType = fuelType,
            segmentsJson = segJson
        )

        serviceScope.launch {
            FuelDatabase.getDatabase(applicationContext).tripDao().insertTrip(trip)
        }
        clearCheckpoint()
    }

    // -------------------------------------------------------------------------
    // Segment helpers
    // -------------------------------------------------------------------------

    private fun buildFinalSegmentsJson(finalEndTime: Long): String {
        val starts = segmentStarts
        if (starts.isEmpty()) {
            val dist = calculateTotalDistance()
            val dur = ((finalEndTime - startTime) / 60_000L).toInt().coerceAtLeast(0)
            return """[{"fuelType":"$activeFuelType","distanceKm":${dist.fmtKm()},"durationMinutes":$dur}]"""
        }

        val sb = StringBuilder("[")
        for (i in starts.indices) {
            val seg = starts[i]
            val endIdx = if (i + 1 < starts.size) starts[i + 1].startRouteIndex else routePoints.size
            val endMs = if (i + 1 < starts.size) starts[i + 1].startTimeMs else finalEndTime

            var dist = 0.0
            val fromIdx = (seg.startRouteIndex + 1).coerceAtMost(endIdx)
            for (j in fromIdx until endIdx) {
                dist += haversineKm(
                    routePoints[j - 1].lat, routePoints[j - 1].lon,
                    routePoints[j].lat, routePoints[j].lon
                )
            }
            val dur = ((endMs - seg.startTimeMs) / 60_000L).toInt().coerceAtLeast(0)

            if (i > 0) sb.append(",")
            sb.append("""{"fuelType":"${seg.fuelType}","distanceKm":${dist.fmtKm()},"durationMinutes":$dur}""")
        }
        sb.append("]")
        return sb.toString()
    }

    private fun Double.fmtKm() = String.format(Locale.US, "%.3f", this)

    // -------------------------------------------------------------------------
    // Checkpoint / recovery
    // -------------------------------------------------------------------------

    private fun saveCheckpoint() {
        if (routePoints.isEmpty()) return
        val prefs = getSharedPreferences(PREFS_CHECKPOINT, Context.MODE_PRIVATE)
        val routeJson = serializeRoute(routePoints)
        val totalDist = calculateTotalDistance()
        val segJson = buildFinalSegmentsJson(System.currentTimeMillis())
        prefs.edit()
            .putBoolean(KEY_HAS_CHECKPOINT, true)
            .putLong(KEY_CP_START_TIME, startTime)
            .putFloat(KEY_CP_START_LAT, startLat.toFloat())
            .putFloat(KEY_CP_START_LON, startLon.toFloat())
            .putString(KEY_CP_ROUTE, routeJson)
            .putFloat(KEY_CP_MAX_SPEED, maxSpeed)
            .putFloat(KEY_CP_DISTANCE, totalDist.toFloat())
            .putString(KEY_CP_FUEL_TYPE, activeFuelType)
            .putString(KEY_CP_SEGMENTS, segJson)
            .apply()
    }

    private fun clearCheckpoint() {
        getSharedPreferences(PREFS_CHECKPOINT, Context.MODE_PRIVATE).edit().clear().apply()
    }

    private fun recoverCheckpointIfExists() {
        val prefs = getSharedPreferences(PREFS_CHECKPOINT, Context.MODE_PRIVATE)
        if (!prefs.getBoolean(KEY_HAS_CHECKPOINT, false)) return

        val distKm = prefs.getFloat(KEY_CP_DISTANCE, 0f).toDouble()
        if (distKm < MIN_TRIP_DISTANCE_KM) {
            clearCheckpoint()
            return
        }

        val cpStartTime = prefs.getLong(KEY_CP_START_TIME, 0L)
        val cpStartLat = prefs.getFloat(KEY_CP_START_LAT, 0f).toDouble()
        val cpStartLon = prefs.getFloat(KEY_CP_START_LON, 0f).toDouble()
        val cpMaxSpeed = prefs.getFloat(KEY_CP_MAX_SPEED, 0f).toDouble()
        val cpRoute = prefs.getString(KEY_CP_ROUTE, "[]") ?: "[]"
        val cpFuelType = prefs.getString(KEY_CP_FUEL_TYPE, "LPG") ?: "LPG"
        val cpSegments = prefs.getString(KEY_CP_SEGMENTS, "[]") ?: "[]"
        val endTime = System.currentTimeMillis()
        val durationMin = ((endTime - cpStartTime) / 60000).toInt()

        val lastPoint = extractLastRoutePoint(cpRoute)
        val endLat = lastPoint?.first ?: cpStartLat
        val endLon = lastPoint?.second ?: cpStartLon
        val avgSpeed = if (durationMin > 0) (distKm / (durationMin / 60.0)) else 0.0

        val trip = TripEntry(
            startTime = cpStartTime,
            endTime = endTime,
            startLat = cpStartLat,
            startLon = cpStartLon,
            endLat = endLat,
            endLon = endLon,
            distanceKm = distKm,
            avgSpeedKmh = avgSpeed,
            maxSpeedKmh = cpMaxSpeed,
            durationMinutes = durationMin,
            routePointsJson = cpRoute,
            fuelType = cpFuelType,
            segmentsJson = cpSegments
        )

        serviceScope.launch {
            FuelDatabase.getDatabase(applicationContext).tripDao().insertTrip(trip)
        }
        clearCheckpoint()
    }

    // -------------------------------------------------------------------------
    // Helper utilities
    // -------------------------------------------------------------------------

    private fun extractLastRoutePoint(json: String): Pair<Double, Double>? {
        val lastBracket = json.lastIndexOf('[')
        if (lastBracket < 0) return null
        val segment = json.substring(lastBracket)
        return try {
            val nums = segment.replace("[", "").replace("]", "").split(",")
            if (nums.size >= 2) Pair(nums[0].trim().toDouble(), nums[1].trim().toDouble())
            else null
        } catch (e: Exception) {
            null
        }
    }

    private fun calculateTotalDistance(): Double {
        var total = 0.0
        for (i in 1 until routePoints.size) {
            total += haversineKm(
                routePoints[i - 1].lat, routePoints[i - 1].lon,
                routePoints[i].lat, routePoints[i].lon
            )
        }
        return total
    }

    private fun haversineKm(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Double {
        val R = 6371.0
        val dLat = Math.toRadians(lat2 - lat1)
        val dLon = Math.toRadians(lon2 - lon1)
        val a = sin(dLat / 2).pow(2) + cos(Math.toRadians(lat1)) * cos(Math.toRadians(lat2)) * sin(dLon / 2).pow(2)
        return R * 2 * asin(sqrt(a))
    }

    private fun serializeRoute(points: List<RoutePoint>): String {
        val sb = StringBuilder("[")
        points.forEachIndexed { i, p ->
            if (i > 0) sb.append(",")
            sb.append("[${p.lat},${p.lon}]")
        }
        sb.append("]")
        return sb.toString()
    }

    private fun currentDistanceKm(): Float {
        if (routePoints.size < 2) return 0f
        var d = 0.0
        for (i in 1 until routePoints.size) {
            d += haversineKm(
                routePoints[i - 1].lat, routePoints[i - 1].lon,
                routePoints[i].lat, routePoints[i].lon
            )
        }
        return d.toFloat()
    }

    private fun broadcastStateUpdate(tripState: TripState, speedKmh: Float) {
        val distanceKm = if (tripState == TripState.RECORDING || tripState == TripState.CONFIRMING_STOP) {
            currentDistanceKm()
        } else 0f

        val intent = Intent(ACTION_TRIP_STATE_UPDATE).apply {
            putExtra(EXTRA_STATE, tripState.name)
            putExtra(EXTRA_SPEED_KMH, speedKmh)
            putExtra(EXTRA_DISTANCE_KM, distanceKm)
        }
        LocalBroadcastManager.getInstance(this).sendBroadcast(intent)
    }

    private fun requestLocationUpdates(slow: Boolean) {
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
            != PackageManager.PERMISSION_GRANTED
        ) return

        try {
            locationManager.removeUpdates(locationListener)
        } catch (e: Exception) {
            // ignore
        }

        try {
            if (slow) {
                locationManager.requestLocationUpdates(
                    LocationManager.GPS_PROVIDER, 5000L, 10f, locationListener
                )
            } else {
                locationManager.requestLocationUpdates(
                    LocationManager.GPS_PROVIDER, 2000L, 5f, locationListener
                )
            }
        } catch (e: SecurityException) {
            // Permission not granted
        } catch (e: Exception) {
            // GPS not available
        }
    }

    private fun buildNotification(): Notification {
        val distanceKm = currentDistanceKm()

        val (title, text, priority) = when (state) {
            TripState.IDLE -> Triple("Yakıt Takibi", "Sürüş bekleniyor...", NotificationCompat.PRIORITY_LOW)
            TripState.CONFIRMING_START -> Triple("Yakıt Takibi", "Hareket tespit edildi...", NotificationCompat.PRIORITY_LOW)
            TripState.RECORDING -> Triple(
                "Sürüş Kaydediliyor",
                String.format("📍 %.1f km • %.0f km/h", distanceKm, if (routePoints.isNotEmpty()) routePoints.last().speedKmh else 0f),
                NotificationCompat.PRIORITY_DEFAULT
            )
            TripState.CONFIRMING_STOP -> Triple("Yakıt Takibi", "Sürüş bitiyor...", NotificationCompat.PRIORITY_LOW)
        }

        val activityIntent = Intent(this, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_SINGLE_TOP
        }
        val pendingIntent = PendingIntent.getActivity(
            this, 0, activityIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle(title)
            .setContentText(text)
            .setSmallIcon(android.R.drawable.ic_menu_mylocation)
            .setOngoing(true)
            .setPriority(priority)
            .setContentIntent(pendingIntent)
            .build()
    }

    private fun updateNotification() {
        val nm = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
        nm.notify(NOTIFICATION_ID, buildNotification())
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID, "Sürüş Takibi", NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "GPS ile otomatik sürüş takibi"
            }
            val nm = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
            nm.createNotificationChannel(channel)
        }
    }
}
