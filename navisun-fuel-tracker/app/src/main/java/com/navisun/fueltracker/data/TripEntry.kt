package com.navisun.fueltracker.data

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "trips")
data class TripEntry(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val startTime: Long,
    val endTime: Long,
    val startLat: Double,
    val startLon: Double,
    val endLat: Double,
    val endLon: Double,
    val distanceKm: Double,
    val avgSpeedKmh: Double,
    val maxSpeedKmh: Double,
    val durationMinutes: Int,
    val routePointsJson: String,  // JSON: [[lat,lon],[lat,lon],...]
    val fuelType: String = "LPG"  // active fuel type during this trip
)
