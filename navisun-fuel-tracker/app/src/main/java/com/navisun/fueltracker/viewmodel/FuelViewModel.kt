package com.navisun.fueltracker.viewmodel

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.viewModelScope
import com.navisun.fueltracker.data.FuelDatabase
import com.navisun.fueltracker.data.FuelEntry
import com.navisun.fueltracker.data.TripDao
import com.navisun.fueltracker.data.TripEntry
import com.navisun.fueltracker.repository.FuelRepository
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

data class FuelTypeStats(
    val fuelType: String,
    val totalEntries: Int = 0,
    val totalCost: Double = 0.0,
    val totalFuel: Double = 0.0,
    val totalKm: Double = 0.0,
    val averageConsumption: Double? = null,
    val bestConsumption: Double? = null,
    val worstConsumption: Double? = null,
    val avgPricePerLiter: Double? = null,
    val avgCostPerKm: Double? = null,
    val lastCostPerKm: Double? = null,
    val recentConsumptions: List<Pair<FuelEntry, Double>> = emptyList()
)

data class FuelStats(
    val totalEntries: Int = 0,
    val averageConsumption: Double? = null,
    val lastConsumption: Double? = null,
    val bestConsumption: Double? = null,
    val worstConsumption: Double? = null,
    val totalCost: Double = 0.0,
    val totalFuel: Double = 0.0,
    val totalKm: Double = 0.0,
    val avgPricePerLiter: Double? = null,
    val avgCostPerKm: Double? = null,
    val lastCostPerKm: Double? = null,
    val recentConsumptions: List<Pair<FuelEntry, Double>> = emptyList(),
    val benzinStats: FuelTypeStats = FuelTypeStats("BENZİN"),
    val lpgStats: FuelTypeStats = FuelTypeStats("LPG")
)

class FuelViewModel(application: Application) : AndroidViewModel(application) {

    private val repository: FuelRepository
    private val tripDao: TripDao

    val allEntries: LiveData<List<FuelEntry>>

    private val _stats = MutableLiveData<FuelStats>()
    val stats: LiveData<FuelStats> = _stats

    init {
        val db = FuelDatabase.getDatabase(application)
        repository = FuelRepository(db.fuelDao())
        tripDao = db.tripDao()
        allEntries = repository.allEntries
    }

    fun insert(entry: FuelEntry) = viewModelScope.launch(Dispatchers.IO) {
        repository.insert(entry)
        refreshStats()
    }

    fun delete(entry: FuelEntry) = viewModelScope.launch(Dispatchers.IO) {
        repository.delete(entry)
        refreshStats()
    }

    fun refreshStats() = viewModelScope.launch(Dispatchers.IO) {
        val entries = repository.getAllEntriesSync()
        val allTrips = tripDao.getAllTripsList()
        val gpsDistances = buildGpsDistanceMap(entries, allTrips)
        val stats = computeStats(entries, gpsDistances, allTrips)
        _stats.postValue(stats)
    }

    // GPS trip mesafelerini ardışık yakıt girişleri arasında hesaplar: entry.id -> km
    private fun buildGpsDistanceMap(
        entriesAsc: List<FuelEntry>,
        allTrips: List<TripEntry>
    ): Map<Long, Double> {
        val result = mutableMapOf<Long, Double>()
        val byType = entriesAsc.sortedBy { it.date }.groupBy { it.fuelType }
        for ((fuelType, typeEntries) in byType) {
            val sorted = typeEntries.sortedBy { it.date }
            for (i in 1 until sorted.size) {
                val prev = sorted[i - 1]
                val curr = sorted[i]
                val totalKm = allTrips
                    .filter { it.startTime >= prev.date && it.endTime <= curr.date }
                    .sumOf { it.getKmForFuelType(fuelType) }
                if (totalKm > 0.3) result[curr.id] = totalKm
            }
        }
        return result
    }

    private fun computeStats(
        entriesAsc: List<FuelEntry>,
        gpsDistances: Map<Long, Double>,
        allTrips: List<TripEntry>
    ): FuelStats {
        if (entriesAsc.isEmpty()) return FuelStats()

        val totalCost = entriesAsc.sumOf { it.fuelAmount * it.pricePerLiter }
        val totalFuel = entriesAsc.sumOf { it.fuelAmount }
        val avgPricePerLiter = totalFuel.takeIf { it > 0 }?.let { totalCost / it }

        val totalKm = allTrips.sumOf { it.distanceKm }
        val avgCostPerKm = if (totalKm > 0) totalCost / totalKm else null

        // Tüketim: fullTank olan ardışık çiftler; GPS mesafesi varsa GPS, yoksa odometer farkı
        val consumptions = mutableListOf<Pair<FuelEntry, Double>>()
        val costPerKmList = mutableListOf<Double>()
        val byType = entriesAsc.groupBy { it.fuelType }
        for ((_, typeEntries) in byType) {
            val sorted = typeEntries.sortedBy { it.date }
            var prevFull: FuelEntry? = null
            for (entry in sorted) {
                if (entry.fullTank) {
                    val prev = prevFull
                    if (prev != null) {
                        val km = gpsDistances[entry.id]
                            ?: (entry.odometer - prev.odometer).takeIf { it > 0 }
                        if (km != null && km > 0) {
                            val c = (entry.fuelAmount / km) * 100.0
                            if (c in 1.0..50.0) {
                                consumptions.add(Pair(entry, c))
                                costPerKmList.add((entry.fuelAmount * entry.pricePerLiter) / km)
                            }
                        }
                    }
                    prevFull = entry
                }
            }
        }

        val benzinEntries = entriesAsc.filter { it.fuelType == "BENZİN" }
        val lpgEntries    = entriesAsc.filter { it.fuelType == "LPG" }

        return FuelStats(
            totalEntries       = entriesAsc.size,
            averageConsumption = consumptions.map { it.second }.average().takeIf { consumptions.isNotEmpty() },
            lastConsumption    = consumptions.lastOrNull()?.second,
            bestConsumption    = consumptions.minOfOrNull { it.second },
            worstConsumption   = consumptions.maxOfOrNull { it.second },
            totalCost          = totalCost,
            totalFuel          = totalFuel,
            totalKm            = totalKm,
            avgPricePerLiter   = avgPricePerLiter,
            avgCostPerKm       = avgCostPerKm,
            lastCostPerKm      = costPerKmList.lastOrNull(),
            recentConsumptions = consumptions.takeLast(5).reversed(),
            benzinStats        = computeFuelTypeStats("BENZİN", benzinEntries, gpsDistances, allTrips),
            lpgStats           = computeFuelTypeStats("LPG", lpgEntries, gpsDistances, allTrips)
        )
    }

    private fun computeFuelTypeStats(
        fuelType: String,
        entriesAsc: List<FuelEntry>,
        gpsDistances: Map<Long, Double>,
        allTrips: List<TripEntry>
    ): FuelTypeStats {
        if (entriesAsc.isEmpty()) return FuelTypeStats(fuelType)

        val totalCost = entriesAsc.sumOf { it.fuelAmount * it.pricePerLiter }
        val totalFuel = entriesAsc.sumOf { it.fuelAmount }
        val avgPricePerLiter = totalFuel.takeIf { it > 0 }?.let { totalCost / it }

        val totalKm = allTrips.sumOf { it.getKmForFuelType(fuelType) }
        val avgCostPerKm = if (totalKm > 0) totalCost / totalKm else null

        val consumptions = mutableListOf<Pair<FuelEntry, Double>>()
        val costPerKmList = mutableListOf<Double>()
        var prevFull: FuelEntry? = null

        for (entry in entriesAsc) {
            if (entry.fullTank) {
                val prev = prevFull
                if (prev != null) {
                    val km = gpsDistances[entry.id]
                        ?: (entry.odometer - prev.odometer).takeIf { it > 0 }
                    if (km != null && km > 0) {
                        val c = (entry.fuelAmount / km) * 100.0
                        if (c in 1.0..50.0) {
                            consumptions.add(Pair(entry, c))
                            costPerKmList.add((entry.fuelAmount * entry.pricePerLiter) / km)
                        }
                    }
                }
                prevFull = entry
            }
        }

        return FuelTypeStats(
            fuelType           = fuelType,
            totalEntries       = entriesAsc.size,
            totalCost          = totalCost,
            totalFuel          = totalFuel,
            totalKm            = totalKm,
            averageConsumption = consumptions.map { it.second }.average().takeIf { consumptions.isNotEmpty() },
            bestConsumption    = consumptions.minOfOrNull { it.second },
            worstConsumption   = consumptions.maxOfOrNull { it.second },
            avgPricePerLiter   = avgPricePerLiter,
            avgCostPerKm       = avgCostPerKm,
            lastCostPerKm      = costPerKmList.lastOrNull(),
            recentConsumptions = consumptions.takeLast(5).reversed()
        )
    }
}
