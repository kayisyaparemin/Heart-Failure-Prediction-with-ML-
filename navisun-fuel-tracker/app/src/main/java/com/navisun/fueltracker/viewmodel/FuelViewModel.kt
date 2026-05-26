package com.navisun.fueltracker.viewmodel

import android.app.Application
import android.content.Context
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
    val averageConsumption: Double? = null
)

data class FuelStats(
    val totalEntries: Int = 0,
    val averageConsumption: Double? = null,  // L/100km
    val lastConsumption: Double? = null,     // L/100km
    val bestConsumption: Double? = null,     // L/100km (en az)
    val worstConsumption: Double? = null,    // L/100km (en fazla)
    val totalCost: Double = 0.0,             // TL
    val totalFuel: Double = 0.0,             // Litre
    val totalKm: Double = 0.0,               // KM
    val avgPricePerLiter: Double? = null,    // TL/L
    val avgCostPerKm: Double? = null,        // TL/km
    val recentConsumptions: List<Pair<FuelEntry, Double>> = emptyList(), // son 5 dolum + tüketim
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
        val initialOdometer = getApplication<Application>()
            .getSharedPreferences("navisun_prefs", Context.MODE_PRIVATE)
            .getFloat("initial_odometer_value", 0f).toDouble()
        val stats = computeStats(entries, trips, initialOdometer)
        _stats.postValue(stats)
    }

    /**
     * Tüketim hesaplama:
     * Sadece fullTank=true olan ardışık iki kayıt arasında hesaplanır.
     * GPS verisi varsa o periyottaki trip mesafeleri kullanılır, yoksa odometer farkı.
     * totalKm için GPS trip toplamı önceliklidir.
     */
    private fun computeStats(
        entriesAsc: List<FuelEntry>,
        allTrips: List<TripEntry>,
        initialOdometer: Double
    ): FuelStats {
        if (entriesAsc.isEmpty()) {
            return FuelStats()
        }

        val totalEntries = entriesAsc.size
        val totalCost = entriesAsc.sumOf { it.fuelAmount * it.pricePerLiter }
        val totalFuel = entriesAsc.sumOf { it.fuelAmount }

        // Toplam km: GPS varsa GPS toplamı, yoksa odometer farkı (başlangıç odometer dahil)
        val totalGpsKm = allTrips.sumOf { it.distanceKm }
        val totalKm = if (totalGpsKm > 0) {
            totalGpsKm
        } else {
            val firstOdo = if (initialOdometer > 0 && entriesAsc.first().odometer > initialOdometer) {
                initialOdometer
            } else {
                entriesAsc.first().odometer
            }
            if (entriesAsc.size >= 2) entriesAsc.last().odometer - firstOdo else 0.0
        }

        val avgPricePerLiter = if (entriesAsc.isNotEmpty()) {
            entriesAsc.sumOf { it.pricePerLiter } / entriesAsc.size
        } else null

        val avgCostPerKm = if (totalKm > 0) totalCost / totalKm else null

        // Tüketim hesaplama: fullTank olan ardışık çiftler
        val consumptions = mutableListOf<Pair<FuelEntry, Double>>()
        var previousFullTankEntry: FuelEntry? = null

        for (entry in entriesAsc) {
            if (entry.fullTank) {
                val prev = previousFullTankEntry
                if (prev != null) {
                    // GPS km bu periyotta (aynı yakıt tipi, tarih aralığı)
                    val gpsKm = allTrips
                        .filter { it.fuelType == entry.fuelType && it.startTime >= prev.date && it.endTime <= entry.date }
                        .sumOf { it.distanceKm }
                    val kmDiff = if (gpsKm > 0) gpsKm else (entry.odometer - prev.odometer)
                    if (kmDiff > 0) {
                        val consumption = (entry.fuelAmount / kmDiff) * 100.0
                        if (consumption in 1.0..50.0) {
                            consumptions.add(Pair(entry, consumption))
                        }
                    }
                }
                previousFullTankEntry = entry
            }
        }

        val avgConsumption = if (consumptions.isNotEmpty()) {
            consumptions.sumOf { it.second } / consumptions.size
        } else null

        val lastConsumption = consumptions.lastOrNull()?.second
        val bestConsumption = consumptions.minOfOrNull { it.second }
        val worstConsumption = consumptions.maxOfOrNull { it.second }
        val recentConsumptions = consumptions.takeLast(5).reversed()

        val benzinEntries = entriesAsc.filter { it.fuelType == "BENZİN" }
        val lpgEntries = entriesAsc.filter { it.fuelType == "LPG" }

        val benzinStats = computeFuelTypeStats("BENZİN", benzinEntries, allTrips)
        val lpgStats = computeFuelTypeStats("LPG", lpgEntries, allTrips)

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
            benzinStats = benzinStats,
            lpgStats = lpgStats
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

        val consumptions = mutableListOf<Double>()
        var previousFullTankEntry: FuelEntry? = null

        for (entry in entriesAsc) {
            if (entry.fullTank) {
                val prev = previousFullTankEntry
                if (prev != null) {
                    val gpsKm = allTrips
                        .filter { it.fuelType == fuelType && it.startTime >= prev.date && it.endTime <= entry.date }
                        .sumOf { it.distanceKm }
                    val kmDiff = if (gpsKm > 0) gpsKm else (entry.odometer - prev.odometer)
                    if (kmDiff > 0) {
                        val consumption = (entry.fuelAmount / kmDiff) * 100.0
                        if (consumption in 1.0..50.0) {
                            consumptions.add(consumption)
                        }
                    }
                }
                previousFullTankEntry = entry
            }
        }

        val avgConsumption = if (consumptions.isNotEmpty()) {
            consumptions.sum() / consumptions.size
        } else null

        return FuelTypeStats(
            fuelType = fuelType,
            totalEntries = entriesAsc.size,
            totalCost = totalCost,
            totalFuel = totalFuel,
            averageConsumption = avgConsumption
        )
    }
}
