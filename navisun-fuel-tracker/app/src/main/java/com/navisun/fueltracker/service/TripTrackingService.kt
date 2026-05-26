package com.navisun.fueltracker.service

import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.content.pm.PackageManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Build
import android.os.Bundle
import android.os.IBinder
import androidx.core.app.ActivityCompat
import androidx.core.app.NotificationCompat
import androidx.localbroadcastmanager.content.LocalBroadcastManager
import com.google.gson.Gson
import com.navisun.fueltracker.R
import com.navisun.fueltracker.data.FuelDatabase
import com.navisun.fueltracker.data.TripEntry
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import kotlin.math.asin
import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

class TripTrackingService : Service() {

    companion object {
        const val ACTION_START = "com.navisun.fueltracker.ACTION_START_TRIP"
        const val ACTION_STOP = "com.navisun.fueltracker.ACTION_STOP_TRIP"
        const val ACTION_SPEED_UPDATE = "SPEED_UPDATE"
        const val EXTRA_SPEED_KMH = "speed_kmh"
        private const val NOTIFICATION_ID = 1001
        private const val CHANNEL_ID = "trip_tracking_channel"
    }

    data class RoutePoint(
        val lat: Double,
        val lon: Double,
        val timestamp: Long,
        val speedKmh: Float
    )

    private val serviceJob = SupervisorJob()
    private val serviceScope = CoroutineScope(Dispatchers.IO + serviceJob)

    private lateinit var locationManager: LocationManager
    private val routePoints = mutableListOf<RoutePoint>()
    private var maxSpeedKmh = 0f
    private var startTime = 0L
    private var totalDistanceKm = 0.0

    private val locationListener = object : LocationListener {
        override fun onLocationChanged(location: Location) {
            val speedKmh = location.speed * 3.6f
            if (speedKmh > maxSpeedKmh) {
                maxSpeedKmh = speedKmh
            }

            val point = RoutePoint(
                lat = location.latitude,
                lon = location.longitude,
                timestamp = location.time,
                speedKmh = speedKmh
            )

            if (routePoints.isNotEmpty()) {
                val prev = routePoints.last()
                val segmentKm = haversineKm(prev.lat, prev.lon, point.lat, point.lon)
                totalDistanceKm += segmentKm
            }

            routePoints.add(point)

            broadcastSpeedUpdate(speedKmh)
            updateNotification(totalDistanceKm)
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
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START -> startTracking()
            ACTION_STOP -> stopTracking()
        }
        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        super.onDestroy()
        serviceJob.cancel()
        try {
            locationManager.removeUpdates(locationListener)
        } catch (e: Exception) {
            // ignore
        }
    }

    private fun startTracking() {
        startTime = System.currentTimeMillis()
        routePoints.clear()
        maxSpeedKmh = 0f
        totalDistanceKm = 0.0

        val notification = buildNotification("Sürüş kaydediliyor... 0.0 km")
        startForeground(NOTIFICATION_ID, notification)

        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
            == PackageManager.PERMISSION_GRANTED
        ) {
            try {
                locationManager.requestLocationUpdates(
                    LocationManager.GPS_PROVIDER,
                    2000L,
                    5f,
                    locationListener
                )
            } catch (e: Exception) {
                // GPS not available
            }
        }
    }

    private fun stopTracking() {
        try {
            locationManager.removeUpdates(locationListener)
        } catch (e: Exception) {
            // ignore
        }

        val endTime = System.currentTimeMillis()
        val durationMs = endTime - startTime
        val durationMinutes = (durationMs / 60000).toInt().coerceAtLeast(1)

        if (routePoints.size >= 2) {
            val avgSpeedKmh = if (durationMs > 0) {
                (totalDistanceKm / (durationMs / 3600000.0))
            } else 0.0

            val startPoint = routePoints.first()
            val endPoint = routePoints.last()

            val routeJson = Gson().toJson(routePoints.map { listOf(it.lat, it.lon) })

            val trip = TripEntry(
                startTime = startTime,
                endTime = endTime,
                startLat = startPoint.lat,
                startLon = startPoint.lon,
                endLat = endPoint.lat,
                endLon = endPoint.lon,
                distanceKm = totalDistanceKm,
                avgSpeedKmh = avgSpeedKmh,
                maxSpeedKmh = maxSpeedKmh.toDouble(),
                durationMinutes = durationMinutes,
                routePointsJson = routeJson
            )

            serviceScope.launch {
                FuelDatabase.getDatabase(applicationContext).tripDao().insertTrip(trip)
            }
        }

        stopForeground(true)
        stopSelf()
    }

    private fun broadcastSpeedUpdate(speedKmh: Float) {
        val intent = Intent(ACTION_SPEED_UPDATE).apply {
            putExtra(EXTRA_SPEED_KMH, speedKmh)
        }
        LocalBroadcastManager.getInstance(this).sendBroadcast(intent)
    }

    private fun updateNotification(distanceKm: Double) {
        val text = String.format("Sürüş kaydediliyor... %.1f km", distanceKm)
        val notification = buildNotification(text)
        val nm = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
        nm.notify(NOTIFICATION_ID, notification)
    }

    private fun buildNotification(text: String): Notification {
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Navisun - Sürüş Kaydı")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.ic_menu_mylocation)
            .setOngoing(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Sürüş Takibi",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "GPS ile sürüş kaydı bildirimleri"
            }
            val nm = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
            nm.createNotificationChannel(channel)
        }
    }

    private fun haversineKm(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Double {
        val R = 6371.0
        val dLat = Math.toRadians(lat2 - lat1)
        val dLon = Math.toRadians(lon2 - lon1)
        val a = sin(dLat / 2).pow(2) +
                cos(Math.toRadians(lat1)) * cos(Math.toRadians(lat2)) * sin(dLon / 2).pow(2)
        return R * 2 * asin(sqrt(a))
    }
}
