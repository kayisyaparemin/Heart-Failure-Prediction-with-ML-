package com.navisun.fueltracker.viewmodel

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.viewModelScope
import com.navisun.fueltracker.data.FuelDatabase
import com.navisun.fueltracker.data.FuelEntry
import com.navisun.fueltracker.repository.FuelRepository
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

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
    val recentConsumptions: List<Pair<FuelEntry, Double>> = emptyList() // son 5 dolum + tüketim
)

class FuelViewModel(application: Application) : AndroidViewModel(application) {

    private val repository: FuelRepository

    val allEntries: LiveData<List<FuelEntry>>

    private val _stats = MutableLiveData<FuelStats>()
    val stats: LiveData<FuelStats> = _stats

    init {
        val fuelDao = FuelDatabase.getDatabase(application).fuelDao()
        repository = FuelRepository(fuelDao)
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
        val entries = repository.getAllEntriesSync() // sorted ASC by date
        val stats = computeStats(entries)
        _stats.postValue(stats)
    }

    /**
     * Tüketim hesaplama:
     * Sadece fullTank=true olan ardışık iki kayıt arasında hesaplanır.
     * consumption = (fuelAmount / (currentOdometer - previousOdometer)) * 100
     */
    private fun computeStats(entriesAsc: List<FuelEntry>): FuelStats {
        if (entriesAsc.isEmpty()) {
            return FuelStats()
        }

        val totalEntries = entriesAsc.size
        val totalCost = entriesAsc.sumOf { it.fuelAmount * it.pricePerLiter }
        val totalFuel = entriesAsc.sumOf { it.fuelAmount }

        // Toplam km: en son - en eski kilometre sayacı
        val totalKm = if (entriesAsc.size >= 2) {
            entriesAsc.last().odometer - entriesAsc.first().odometer
        } else {
            0.0
        }

        // Ortalama litre fiyatı
        val avgPricePerLiter = if (entriesAsc.isNotEmpty()) {
            entriesAsc.sumOf { it.pricePerLiter } / entriesAsc.size
        } else null

        // Tüketim hesaplama: fullTank olan ardışık çiftler
        val consumptions = mutableListOf<Pair<FuelEntry, Double>>()
        var previousFullTankEntry: FuelEntry? = null

        for (entry in entriesAsc) {
            if (entry.fullTank) {
                val prev = previousFullTankEntry
                if (prev != null) {
                    val kmDiff = entry.odometer - prev.odometer
                    if (kmDiff > 0) {
                        val consumption = (entry.fuelAmount / kmDiff) * 100.0
                        if (consumption in 1.0..50.0) { // Makul aralık kontrolü
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

        // Son 5 tüketim (en yeniden en eskiye)
        val recentConsumptions = consumptions.takeLast(5).reversed()

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
            recentConsumptions = recentConsumptions
        )
    }
}
