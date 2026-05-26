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
    val routePointsJson: String,
    val fuelType: String = "LPG",
    val segmentsJson: String = "[]"  // JSON: [{"fuelType":"LPG","distanceKm":12.3,"durationMinutes":15},...]
) {
    /** GPS km for a specific fuel type, using segment data when available. */
    fun getKmForFuelType(type: String): Double {
        return try {
            val arr = org.json.JSONArray(segmentsJson)
            if (arr.length() == 0) return if (fuelType == type) distanceKm else 0.0
            var total = 0.0
            for (i in 0 until arr.length()) {
                val obj = arr.getJSONObject(i)
                if (obj.optString("fuelType") == type) total += obj.optDouble("distanceKm", 0.0)
            }
            total
        } catch (e: Exception) {
            if (fuelType == type) distanceKm else 0.0
        }
    }
}
