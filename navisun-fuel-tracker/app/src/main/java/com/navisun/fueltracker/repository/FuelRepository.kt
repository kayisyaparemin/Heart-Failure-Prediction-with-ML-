package com.navisun.fueltracker.repository

import androidx.lifecycle.LiveData
import com.navisun.fueltracker.data.FuelDao
import com.navisun.fueltracker.data.FuelEntry

class FuelRepository(private val fuelDao: FuelDao) {

    val allEntries: LiveData<List<FuelEntry>> = fuelDao.getAllEntries()

    suspend fun insert(entry: FuelEntry): Long {
        return fuelDao.insertFuelEntry(entry)
    }

    suspend fun delete(entry: FuelEntry) {
        fuelDao.deleteFuelEntry(entry)
    }

    suspend fun getLastEntry(): FuelEntry? {
        return fuelDao.getLastEntry()
    }

    suspend fun getLastEntries(limit: Int): List<FuelEntry> {
        return fuelDao.getLastEntries(limit)
    }

    suspend fun getEntryCount(): Int {
        return fuelDao.getEntryCount()
    }

    suspend fun getAllEntriesSync(): List<FuelEntry> {
        return fuelDao.getAllEntriesSync()
    }
}
