package com.navisun.fueltracker.data

import androidx.lifecycle.LiveData
import androidx.room.Dao
import androidx.room.Delete
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query

@Dao
interface TripDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertTrip(trip: TripEntry): Long

    @Delete
    suspend fun deleteTrip(trip: TripEntry)

    @Query("SELECT * FROM trips ORDER BY startTime DESC")
    fun getAllTrips(): LiveData<List<TripEntry>>

    @Query("SELECT * FROM trips WHERE id = :tripId LIMIT 1")
    suspend fun getTripById(tripId: Long): TripEntry?
}
