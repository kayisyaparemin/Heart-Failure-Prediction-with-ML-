package com.navisun.fueltracker.viewmodel

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.viewModelScope
import com.navisun.fueltracker.data.FuelDatabase
import com.navisun.fueltracker.data.TripEntry
import com.navisun.fueltracker.repository.TripRepository
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

class TripViewModel(application: Application) : AndroidViewModel(application) {

    private val repository: TripRepository

    val allTrips: LiveData<List<TripEntry>>

    init {
        val tripDao = FuelDatabase.getDatabase(application).tripDao()
        repository = TripRepository(tripDao)
        allTrips = repository.allTrips
    }

    fun deleteTrip(trip: TripEntry) = viewModelScope.launch(Dispatchers.IO) {
        repository.deleteTrip(trip)
    }
}
