package com.navisun.fueltracker.data

import androidx.lifecycle.LiveData
import androidx.room.Dao
import androidx.room.Delete
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query

@Dao
interface FuelDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertFuelEntry(entry: FuelEntry): Long

    @Delete
    suspend fun deleteFuelEntry(entry: FuelEntry)

    @Query("SELECT * FROM fuel_entries ORDER BY date DESC")
    fun getAllEntries(): LiveData<List<FuelEntry>>

    @Query("SELECT * FROM fuel_entries ORDER BY date DESC LIMIT 1")
    suspend fun getLastEntry(): FuelEntry?

    @Query("SELECT * FROM fuel_entries ORDER BY date DESC LIMIT :limit")
    suspend fun getLastEntries(limit: Int): List<FuelEntry>

    @Query("SELECT COUNT(*) FROM fuel_entries")
    suspend fun getEntryCount(): Int

    @Query("SELECT * FROM fuel_entries ORDER BY date ASC")
    suspend fun getAllEntriesSync(): List<FuelEntry>
}
