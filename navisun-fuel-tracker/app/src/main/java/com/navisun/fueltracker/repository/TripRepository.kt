package com.navisun.fueltracker.repository

import androidx.lifecycle.LiveData
import com.navisun.fueltracker.data.TripDao
import com.navisun.fueltracker.data.TripEntry

class TripRepository(private val tripDao: TripDao) {

    val allTrips: LiveData<List<TripEntry>> = tripDao.getAllTrips()

    suspend fun insertTrip(trip: TripEntry): Long {
        return tripDao.insertTrip(trip)
    }

    suspend fun deleteTrip(trip: TripEntry) {
        tripDao.deleteTrip(trip)
    }

    suspend fun getTripById(tripId: Long): TripEntry? {
        return tripDao.getTripById(tripId)
    }
}
