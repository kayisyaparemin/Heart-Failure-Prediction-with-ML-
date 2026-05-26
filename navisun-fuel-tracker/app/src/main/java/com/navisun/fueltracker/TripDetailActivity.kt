package com.navisun.fueltracker

import android.os.Bundle
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

    private lateinit var binding: ActivityTripDetailBinding
    private val dateFormat = SimpleDateFormat("dd.MM.yyyy HH:mm", Locale("tr", "TR"))

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // OSMDroid configuration must be set before MapView is inflated
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
        // Stats
        binding.tvDetailDate.text = dateFormat.format(Date(trip.startTime))
        binding.tvDetailDistance.text = String.format("%.1f km", trip.distanceKm)
        binding.tvDetailDuration.text = String.format("%d dk", trip.durationMinutes)
        binding.tvDetailAvgSpeed.text = String.format("%.0f km/s", trip.avgSpeedKmh)
        binding.tvDetailMaxSpeed.text = String.format("%.0f km/s", trip.maxSpeedKmh)

        // Map
        setupRouteOnMap(trip)
    }

    private fun setupRouteOnMap(trip: TripEntry) {
        val overlays = binding.mapView.overlays
        overlays.clear()

        // Parse route points from JSON
        val routePoints = parseRoutePoints(trip.routePointsJson)

        if (routePoints.isEmpty()) {
            // Just show start/end markers if no route
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

        // Draw polyline
        val polyline = Polyline().apply {
            setPoints(geoPoints)
            outlinePaint.color = android.graphics.Color.parseColor("#2196F3")
            outlinePaint.strokeWidth = 8f
        }
        overlays.add(polyline)

        // Start marker (green)
        val startPoint = geoPoints.first()
        addMarker(startPoint, "Başlangıç", isStart = true)

        // End marker (red)
        val endPoint = geoPoints.last()
        addMarker(endPoint, "Bitiş", isStart = false)

        // Auto-zoom to fit bounds
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
                    binding.mapView.controller.setCenter(startPoint)
                    binding.mapView.controller.setZoom(13.0)
                }
            }
        } else {
            binding.mapView.controller.setCenter(startPoint)
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
                resources.getDrawable(android.R.drawable.presence_online, null).apply {
                    setColorFilter(
                        android.graphics.Color.GREEN,
                        android.graphics.PorterDuff.Mode.SRC_IN
                    )
                }
            } else {
                resources.getDrawable(android.R.drawable.presence_busy, null).apply {
                    setColorFilter(
                        android.graphics.Color.RED,
                        android.graphics.PorterDuff.Mode.SRC_IN
                    )
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
