package com.navisun.fueltracker

import android.content.Context
import com.navisun.fueltracker.data.FuelDatabase
import com.navisun.fueltracker.data.FuelEntry
import com.navisun.fueltracker.data.TripEntry
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.concurrent.TimeUnit

object TestDataHelper {

    private val DAY = TimeUnit.DAYS.toMillis(1)
    private val HOUR = TimeUnit.HOURS.toMillis(1)
    private val MIN = TimeUnit.MINUTES.toMillis(1)

    suspend fun insertTestData(context: Context) = withContext(Dispatchers.IO) {
        val db = FuelDatabase.getDatabase(context)
        val now = System.currentTimeMillis()

        // --- SÜRÜŞLER ---
        // Zaman çizelgesi:
        //  9 gün önce  : LPG baz dolum
        //  8 gün önce  : Sürüş 1 – LPG, 24.8km (Kadıköy → Beşiktaş)
        //  7 gün önce  : BENZİN baz dolum
        //  6 gün önce  : Sürüş 2 – LPG→BENZİN geçiş, LPG:10.2km BENZİN:8.1km
        //  5 gün önce  : LPG dolum 2 (35km için 4.0L → 11.4 L/100km)
        //  4 gün önce  : Sürüş 3 – BENZİN, 12.1km (Üsküdar → Bostancı)
        //  2 gün önce  : Sürüş 4 – LPG, 29.6km (uzun yol)
        //  1 gün önce  : Sürüş 5 – LPG, 8.4km (kısa gidiş)
        //  bugün -2sa  : LPG dolum 3 (38km için 4.5L → 11.8 L/100km)
        //  bugün -1sa  : BENZİN dolum 2 (20.2km için 2.0L → 9.9 L/100km)

        val trips = listOf(
            // Sürüş 1: LPG, Kadıköy → Beşiktaş çevresi
            TripEntry(
                startTime  = now - 8 * DAY,
                endTime    = now - 8 * DAY + 35 * MIN,
                startLat   = 40.9900, startLon = 29.0280,
                endLat     = 41.0430, endLon   = 29.0080,
                distanceKm = 24.8,
                avgSpeedKmh = 42.5,
                maxSpeedKmh = 86.0,
                durationMinutes = 35,
                routePointsJson = interpolateRoute(40.9900, 29.0280, 41.0430, 29.0080, 14),
                fuelType    = "LPG",
                segmentsJson = """[{"fuelType":"LPG","distanceKm":24.800,"durationMinutes":35}]"""
            ),
            // Sürüş 2: Yarı LPG yarı BENZİN (yakıt değişikliği testi)
            TripEntry(
                startTime  = now - 6 * DAY,
                endTime    = now - 6 * DAY + 28 * MIN,
                startLat   = 41.0430, startLon = 29.0080,
                endLat     = 41.0650, endLon   = 29.0380,
                distanceKm = 18.3,
                avgSpeedKmh = 39.2,
                maxSpeedKmh = 74.0,
                durationMinutes = 28,
                routePointsJson = interpolateRoute(41.0430, 29.0080, 41.0650, 29.0380, 10),
                fuelType    = "BENZİN",
                segmentsJson = """[{"fuelType":"LPG","distanceKm":10.200,"durationMinutes":15},{"fuelType":"BENZİN","distanceKm":8.100,"durationMinutes":13}]"""
            ),
            // Sürüş 3: BENZİN, Üsküdar → Bostancı
            TripEntry(
                startTime  = now - 4 * DAY,
                endTime    = now - 4 * DAY + 22 * MIN,
                startLat   = 41.0230, startLon = 29.0130,
                endLat     = 40.9630, endLon   = 29.0980,
                distanceKm = 12.1,
                avgSpeedKmh = 33.0,
                maxSpeedKmh = 62.0,
                durationMinutes = 22,
                routePointsJson = interpolateRoute(41.0230, 29.0130, 40.9630, 29.0980, 8),
                fuelType    = "BENZİN",
                segmentsJson = """[{"fuelType":"BENZİN","distanceKm":12.100,"durationMinutes":22}]"""
            ),
            // Sürüş 4: LPG, uzun yol (E5 güzergahı)
            TripEntry(
                startTime  = now - 2 * DAY,
                endTime    = now - 2 * DAY + 45 * MIN,
                startLat   = 40.9630, startLon = 29.0980,
                endLat     = 40.9200, endLon   = 28.8200,
                distanceKm = 29.6,
                avgSpeedKmh = 39.5,
                maxSpeedKmh = 94.0,
                durationMinutes = 45,
                routePointsJson = interpolateRoute(40.9630, 29.0980, 40.9200, 28.8200, 16),
                fuelType    = "LPG",
                segmentsJson = """[{"fuelType":"LPG","distanceKm":29.600,"durationMinutes":45}]"""
            ),
            // Sürüş 5: LPG, kısa dönüş
            TripEntry(
                startTime  = now - 1 * DAY,
                endTime    = now - 1 * DAY + 15 * MIN,
                startLat   = 40.9200, startLon = 28.8200,
                endLat     = 40.9900, endLon   = 29.0280,
                distanceKm = 8.4,
                avgSpeedKmh = 33.6,
                maxSpeedKmh = 58.0,
                durationMinutes = 15,
                routePointsJson = interpolateRoute(40.9200, 28.8200, 40.9900, 29.0280, 6),
                fuelType    = "LPG",
                segmentsJson = """[{"fuelType":"LPG","distanceKm":8.400,"durationMinutes":15}]"""
            )
        )

        trips.forEach { db.tripDao().insertTrip(it) }

        // --- YAKIT KAYITLARI ---
        // Tüketim hesabı için fullTank=true olan ardışık çiftler kullanılır.
        // GPS km her yakıt tipi için segmentJson üzerinden gelir.
        val fuelEntries = listOf(
            // LPG baz dolum (9 gün önce) — bu önceki referans noktası
            FuelEntry(date = now - 9 * DAY, odometer = 0.0,
                fuelAmount = 35.0, pricePerLiter = 8.50,
                fullTank = true, note = "Test – LPG baz dolum", fuelType = "LPG"),
            // BENZİN baz dolum (7 gün önce)
            FuelEntry(date = now - 7 * DAY, odometer = 0.0,
                fuelAmount = 2.5, pricePerLiter = 32.50,
                fullTank = true, note = "Test – BENZİN baz dolum", fuelType = "BENZİN"),
            // LPG dolum 2 (5 gün önce) → Sürüş 1 LPG(24.8) + Sürüş 2 LPG(10.2) = 35km
            // Tüketim: 4.0L / 35km × 100 = 11.4 L/100km
            FuelEntry(date = now - 5 * DAY, odometer = 0.0,
                fuelAmount = 4.0, pricePerLiter = 8.75,
                fullTank = true, note = "Test – LPG dolum", fuelType = "LPG"),
            // LPG dolum 3 (bugün -2sa) → Sürüş 4 LPG(29.6) + Sürüş 5 LPG(8.4) = 38km
            // Tüketim: 4.5L / 38km × 100 = 11.8 L/100km
            FuelEntry(date = now - 2 * HOUR, odometer = 0.0,
                fuelAmount = 4.5, pricePerLiter = 8.80,
                fullTank = true, note = "Test – LPG dolum", fuelType = "LPG"),
            // BENZİN dolum 2 (bugün -1sa) → Sürüş 2 BENZİN(8.1) + Sürüş 3 BENZİN(12.1) = 20.2km
            // Tüketim: 2.0L / 20.2km × 100 = 9.9 L/100km
            FuelEntry(date = now - 1 * HOUR, odometer = 0.0,
                fuelAmount = 2.0, pricePerLiter = 33.00,
                fullTank = true, note = "Test – BENZİN dolum", fuelType = "BENZİN")
        )

        fuelEntries.forEach { db.fuelDao().insertFuelEntry(it) }
    }

    suspend fun clearAllData(context: Context) = withContext(Dispatchers.IO) {
        val db = FuelDatabase.getDatabase(context)
        db.clearAllTables()
    }

    // İki nokta arasında doğrusal interpolasyon ile rota üretir
    private fun interpolateRoute(
        startLat: Double, startLon: Double,
        endLat: Double, endLon: Double,
        steps: Int
    ): String {
        val sb = StringBuilder("[")
        for (i in 0 until steps) {
            val t = i.toDouble() / (steps - 1).coerceAtLeast(1)
            val lat = startLat + (endLat - startLat) * t
            val lon = startLon + (endLon - startLon) * t
            if (i > 0) sb.append(",")
            sb.append("[$lat,$lon]")
        }
        sb.append("]")
        return sb.toString()
    }
}
