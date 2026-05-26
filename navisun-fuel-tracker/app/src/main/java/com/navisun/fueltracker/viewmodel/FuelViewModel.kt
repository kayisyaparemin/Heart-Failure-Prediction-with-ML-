package com.navisun.fueltracker.viewmodel

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.viewModelScope
import com.navisun.fueltracker.data.FuelDatabase
import com.navisun.fueltracker.data.FuelEntry
import com.navisun.fueltracker.data.TripEntry
import com.navisun.fueltracker.repository.FuelRepository
import com.navisun.fueltracker.repository.TripRepository
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
    val recentConsumptions: List<Pair<FuelEntry, Double>> = emptyList(),
    val benzinStats: FuelTypeStats = FuelTypeStats("BENZİN"),
    val lpgStats: FuelTypeStats = FuelTypeStats("LPG")
)

class FuelViewModel(application: Application) : AndroidViewModel(application) {

    private val repository: FuelRepository
    private val tripRepository: TripRepository

    val allEntries: LiveData<List<FuelEntry>>

    private val _stats = MutableLiveData<FuelStats>()
    val stats: LiveData<FuelStats> = _stats

    init {
        val db = FuelDatabase.getDatabase(application)
        repository = FuelRepository(db.fuelDao())
        tripRepository = TripRepository(db.tripDao())
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
        val trips = tripRepository.getAllTripsList()
        _stats.postValue(computeStats(entries, trips))
    }

    private fun computeStats(entriesAsc: List<FuelEntry>, allTrips: List<TripEntry>): FuelStats {
        if (entriesAsc.isEmpty()) return FuelStats()

        val totalEntries = entriesAsc.size
        val totalCost = entriesAsc.sumOf { it.fuelAmount * it.pricePerLiter }
        val totalFuel = entriesAsc.sumOf { it.fuelAmount }

        // Toplam km: GPS toplamı (tüm yakıt tipleri dahil), fallback odometer
        val totalGpsKm = allTrips.sumOf { it.distanceKm }
        val totalKm = if (totalGpsKm > 0) totalGpsKm
        else if (entriesAsc.size >= 2) entriesAsc.last().odometer - entriesAsc.first().odometer
        else 0.0

        val avgPricePerLiter = entriesAsc.sumOf { it.pricePerLiter } / entriesAsc.size
        val avgCostPerKm = if (totalKm > 0) totalCost / totalKm else null

        // Tüketim: fullTank olan ardışık çiftler, GPS km (segment bazlı) öncelikli
        val consumptions = mutableListOf<Pair<FuelEntry, Double>>()
        var previousFullTankEntry: FuelEntry? = null

        for (entry in entriesAsc) {
            if (entry.fullTank) {
                val prev = previousFullTankEntry
                if (prev != null) {
                    val gpsKm = allTrips
                        .filter { it.startTime >= prev.date && it.endTime <= entry.date }
                        .sumOf { it.getKmForFuelType(entry.fuelType) }
                    val kmDiff = if (gpsKm > 0) gpsKm else (entry.odometer - prev.odometer)
                    if (kmDiff > 0) {
                        val consumption = (entry.fuelAmount / kmDiff) * 100.0
                        if (consumption in 1.0..50.0) consumptions.add(Pair(entry, consumption))
                    }
                }
                previousFullTankEntry = entry
            }
        }

        val avgConsumption = if (consumptions.isNotEmpty()) consumptions.sumOf { it.second } / consumptions.size else null
        val lastConsumption = consumptions.lastOrNull()?.second
        val bestConsumption = consumptions.minOfOrNull { it.second }
        val worstConsumption = consumptions.maxOfOrNull { it.second }
        val recentConsumptions = consumptions.takeLast(5).reversed()

        val benzinEntries = entriesAsc.filter { it.fuelType == "BENZİN" }
        val lpgEntries = entriesAsc.filter { it.fuelType == "LPG" }

        return FuelStats(
            totalEntries = totalEntries,
            averageConsumption = avgConsumption,
            lastConsumption = lastConsumption,
            bestConsumption = bestConsumption,
            worstConsumption = worstConsumption,
            totalCost = totalCost,
            totalFuel = totalFuel,
            totalKm = totalKm,
            avgPricePerLiter = avgPricePerLiter,
            avgCostPerKm = avgCostPerKm,
            recentConsumptions = recentConsumptions,
            benzinStats = computeFuelTypeStats("BENZİN", benzinEntries, allTrips),
            lpgStats = computeFuelTypeStats("LPG", lpgEntries, allTrips)
        )
    }

    private fun computeFuelTypeStats(
        fuelType: String,
        entriesAsc: List<FuelEntry>,
        allTrips: List<TripEntry>
    ): FuelTypeStats {
        if (entriesAsc.isEmpty()) return FuelTypeStats(fuelType)

        val totalCost = entriesAsc.sumOf { it.fuelAmount * it.pricePerLiter }
        val totalFuel = entriesAsc.sumOf { it.fuelAmount }
        val totalKm = allTrips.sumOf { it.getKmForFuelType(fuelType) }
        val avgPricePerLiter = entriesAsc.sumOf { it.pricePerLiter } / entriesAsc.size
        val avgCostPerKm = if (totalKm > 0) totalCost / totalKm else null

        val consumptions = mutableListOf<Pair<FuelEntry, Double>>()
        var previousFullTankEntry: FuelEntry? = null

        for (entry in entriesAsc) {
            if (entry.fullTank) {
                val prev = previousFullTankEntry
                if (prev != null) {
                    val gpsKm = allTrips
                        .filter { it.startTime >= prev.date && it.endTime <= entry.date }
                        .sumOf { it.getKmForFuelType(fuelType) }
                    val kmDiff = if (gpsKm > 0) gpsKm else (entry.odometer - prev.odometer)
                    if (kmDiff > 0) {
                        val consumption = (entry.fuelAmount / kmDiff) * 100.0
                        if (consumption in 1.0..50.0) consumptions.add(Pair(entry, consumption))
                    }
                }
                previousFullTankEntry = entry
            }
        }

        val avgConsumption = if (consumptions.isNotEmpty()) consumptions.sumOf { it.second } / consumptions.size else null

        return FuelTypeStats(
            fuelType = fuelType,
            totalEntries = entriesAsc.size,
            totalCost = totalCost,
            totalFuel = totalFuel,
            totalKm = totalKm,
            averageConsumption = avgConsumption,
            bestConsumption = consumptions.minOfOrNull { it.second },
            worstConsumption = consumptions.maxOfOrNull { it.second },
            avgPricePerLiter = avgPricePerLiter,
            avgCostPerKm = avgCostPerKm,
            recentConsumptions = consumptions.takeLast(5).reversed()
        )
    }
}
