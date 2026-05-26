package com.navisun.fueltracker

import android.graphics.Color
import android.os.Bundle
import android.view.View
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import android.content.Context
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import com.navisun.fueltracker.data.FuelDatabase
import com.navisun.fueltracker.data.TripEntry
import com.navisun.fueltracker.databinding.ActivityTripDetailBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.osmdroid.config.Configuration
import org.osmdroid.tileprovider.tilesource.TileSourceFactory
import org.osmdroid.util.BoundingBox
import org.osmdroid.util.GeoPoint
import org.osmdroid.views.overlay.Marker
import org.osmdroid.views.overlay.Polyline
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class TripDetailActivity : AppCompatActivity() {

    companion object {
        const val EXTRA_TRIP_ID = "extra_trip_id"
    }

    private data class TripSegment(
        val fuelType: String = "LPG",
        val distanceKm: Double = 0.0,
        val durationMinutes: Int = 0
    )

    private lateinit var binding: ActivityTripDetailBinding
    private val dateFormat = SimpleDateFormat("dd.MM.yyyy HH:mm", Locale("tr", "TR"))

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Configuration.getInstance().load(
            applicationContext,
            applicationContext.getSharedPreferences("osmdroid", Context.MODE_PRIVATE)
        )
        Configuration.getInstance().userAgentValue = packageName

        binding = ActivityTripDetailBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupToolbar()
        setupMap()

        val tripId = intent.getLongExtra(EXTRA_TRIP_ID, -1L)
        if (tripId != -1L) {
            loadTrip(tripId)
        } else {
            finish()
        }
    }

    override fun onResume() {
        super.onResume()
        binding.mapView.onResume()
    }

    override fun onPause() {
        super.onPause()
        binding.mapView.onPause()
    }

    private fun setupToolbar() {
        setSupportActionBar(binding.toolbar)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        supportActionBar?.title = getString(R.string.trip_detail)
        binding.toolbar.setNavigationOnClickListener { finish() }
    }

    private fun setupMap() {
        binding.mapView.setTileSource(TileSourceFactory.MAPNIK)
        binding.mapView.setMultiTouchControls(true)
        binding.mapView.controller.setZoom(13.0)
    }

    private fun loadTrip(tripId: Long) {
        lifecycleScope.launch {
            val trip = withContext(Dispatchers.IO) {
                FuelDatabase.getDatabase(applicationContext).tripDao().getTripById(tripId)
            }
            trip?.let { displayTrip(it) }
        }
    }

    private fun displayTrip(trip: TripEntry) {
        binding.tvDetailDate.text = dateFormat.format(Date(trip.startTime))
        binding.tvDetailDistance.text = String.format("%.1f km", trip.distanceKm)
        binding.tvDetailDuration.text = String.format("%d dk", trip.durationMinutes)
        binding.tvDetailAvgSpeed.text = String.format("%.0f km/s", trip.avgSpeedKmh)
        binding.tvDetailMaxSpeed.text = String.format("%.0f km/s", trip.maxSpeedKmh)

        // Yakıt tipi rozeti
        binding.tvDetailFuelType.text = trip.fuelType
        val badgeColor = if (trip.fuelType == "LPG") Color.parseColor("#00897b") else Color.parseColor("#f57c00")
        val drawable = androidx.core.content.ContextCompat.getDrawable(this, R.drawable.bg_fuel_badge)?.mutate()
        (drawable as? android.graphics.drawable.GradientDrawable)?.setColor(badgeColor)
        binding.tvDetailFuelType.background = drawable

        displaySegments(trip)
        setupRouteOnMap(trip)
    }

    private fun displaySegments(trip: TripEntry) {
        val segments = parseSegments(trip.segmentsJson)

        // Sadece birden fazla segment varsa bölümü göster
        if (segments.size <= 1) {
            binding.containerSegments.visibility = View.GONE
            return
        }

        binding.containerSegments.visibility = View.VISIBLE
        binding.llSegments.removeAllViews()

        for (seg in segments) {
            val row = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
                ).also { it.bottomMargin = 8.dp }
                gravity = android.view.Gravity.CENTER_VERTICAL
            }

            // Yakıt rozeti
            val badge = TextView(this).apply {
                text = seg.fuelType
                setTextColor(Color.WHITE)
                textSize = 12f
                setTypeface(null, android.graphics.Typeface.BOLD)
                val badgeColor = if (seg.fuelType == "LPG") Color.parseColor("#00897b") else Color.parseColor("#f57c00")
                val bg = androidx.core.content.ContextCompat.getDrawable(this@TripDetailActivity, R.drawable.bg_fuel_badge)?.mutate()
                (bg as? android.graphics.drawable.GradientDrawable)?.setColor(badgeColor)
                background = bg
                setPadding(20, 8, 20, 8)
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.WRAP_CONTENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
                ).also { it.marginEnd = 24.dp }
            }

            // Mesafe
            val distance = TextView(this).apply {
                text = String.format("%.1f km", seg.distanceKm)
                setTextColor(androidx.core.content.ContextCompat.getColor(this@TripDetailActivity, R.color.text_primary))
                textSize = 15f
                setTypeface(null, android.graphics.Typeface.BOLD)
                layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
            }

            // Süre
            val duration = TextView(this).apply {
                text = "${seg.durationMinutes} dk"
                setTextColor(androidx.core.content.ContextCompat.getColor(this@TripDetailActivity, R.color.text_secondary))
                textSize = 14f
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.WRAP_CONTENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
                )
            }

            row.addView(badge)
            row.addView(distance)
            row.addView(duration)
            binding.llSegments.addView(row)
        }
    }

    private val Int.dp: Int get() = (this * resources.displayMetrics.density).toInt()

    private fun parseSegments(json: String): List<TripSegment> {
        return try {
            val type = object : TypeToken<List<TripSegment>>() {}.type
            Gson().fromJson(json, type) ?: emptyList()
        } catch (e: Exception) {
            emptyList()
        }
    }

    private fun setupRouteOnMap(trip: TripEntry) {
        val overlays = binding.mapView.overlays
        overlays.clear()

        val routePoints = parseRoutePoints(trip.routePointsJson)

        if (routePoints.isEmpty()) {
            if (trip.startLat != 0.0 || trip.startLon != 0.0) {
                val startPoint = GeoPoint(trip.startLat, trip.startLon)
                val endPoint = GeoPoint(trip.endLat, trip.endLon)
                addMarker(startPoint, "Başlangıç", isStart = true)
                addMarker(endPoint, "Bitiş", isStart = false)
                binding.mapView.controller.setCenter(startPoint)
            }
            return
        }

        val geoPoints = routePoints.map { GeoPoint(it[0], it[1]) }

        val polyline = Polyline().apply {
            setPoints(geoPoints)
            outlinePaint.color = Color.parseColor("#2196F3")
            outlinePaint.strokeWidth = 8f
        }
        overlays.add(polyline)

        addMarker(geoPoints.first(), "Başlangıç", isStart = true)
        addMarker(geoPoints.last(), "Bitiş", isStart = false)

        if (geoPoints.size >= 2) {
            val minLat = geoPoints.minOf { it.latitude }
            val maxLat = geoPoints.maxOf { it.latitude }
            val minLon = geoPoints.minOf { it.longitude }
            val maxLon = geoPoints.maxOf { it.longitude }
            val boundingBox = BoundingBox(maxLat, maxLon, minLat, minLon)
            binding.mapView.post {
                try {
                    binding.mapView.zoomToBoundingBox(boundingBox, true, 80)
                } catch (e: Exception) {
                    binding.mapView.controller.setCenter(geoPoints.first())
                    binding.mapView.controller.setZoom(13.0)
                }
            }
        } else {
            binding.mapView.controller.setCenter(geoPoints.first())
            binding.mapView.controller.setZoom(13.0)
        }

        binding.mapView.invalidate()
    }

    private fun addMarker(point: GeoPoint, title: String, isStart: Boolean) {
        val marker = Marker(binding.mapView).apply {
            position = point
            setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_BOTTOM)
            this.title = title
            icon = if (isStart) {
                androidx.core.content.ContextCompat.getDrawable(
                    this@TripDetailActivity, android.R.drawable.presence_online
                )?.mutate()?.also {
                    it.setColorFilter(Color.GREEN, android.graphics.PorterDuff.Mode.SRC_IN)
                }
            } else {
                androidx.core.content.ContextCompat.getDrawable(
                    this@TripDetailActivity, android.R.drawable.presence_busy
                )?.mutate()?.also {
                    it.setColorFilter(Color.RED, android.graphics.PorterDuff.Mode.SRC_IN)
                }
            }
        }
        binding.mapView.overlays.add(marker)
    }

    private fun parseRoutePoints(json: String): List<List<Double>> {
        return try {
            val type = object : TypeToken<List<List<Double>>>() {}.type
            Gson().fromJson(json, type) ?: emptyList()
        } catch (e: Exception) {
            emptyList()
        }
    }
}
