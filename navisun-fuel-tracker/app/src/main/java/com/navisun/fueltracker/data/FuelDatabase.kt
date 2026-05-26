package com.navisun.fueltracker.data

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase

@Database(
    entities = [FuelEntry::class, TripEntry::class],
    version = 4,
    exportSchema = false
)
abstract class FuelDatabase : RoomDatabase() {

    abstract fun fuelDao(): FuelDao
    abstract fun tripDao(): TripDao

    companion object {
        @Volatile
        private var INSTANCE: FuelDatabase? = null

        val MIGRATION_1_2 = object : Migration(1, 2) {
            override fun migrate(database: SupportSQLiteDatabase) {
                database.execSQL("ALTER TABLE fuel_entries ADD COLUMN fuelType TEXT NOT NULL DEFAULT 'BENZİN'")
                database.execSQL("""CREATE TABLE IF NOT EXISTS trips (
                    id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                    startTime INTEGER NOT NULL,
                    endTime INTEGER NOT NULL,
                    startLat REAL NOT NULL,
                    startLon REAL NOT NULL,
                    endLat REAL NOT NULL,
                    endLon REAL NOT NULL,
                    distanceKm REAL NOT NULL,
                    avgSpeedKmh REAL NOT NULL,
                    maxSpeedKmh REAL NOT NULL,
                    durationMinutes INTEGER NOT NULL,
                    routePointsJson TEXT NOT NULL
                )""")
            }
        }

        val MIGRATION_2_3 = object : Migration(2, 3) {
            override fun migrate(database: SupportSQLiteDatabase) {
                database.execSQL("ALTER TABLE trips ADD COLUMN fuelType TEXT NOT NULL DEFAULT 'LPG'")
            }
        }

        val MIGRATION_3_4 = object : Migration(3, 4) {
            override fun migrate(database: SupportSQLiteDatabase) {
                database.execSQL("ALTER TABLE trips ADD COLUMN segmentsJson TEXT NOT NULL DEFAULT '[]'")
            }
        }

        fun getDatabase(context: Context): FuelDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    FuelDatabase::class.java,
                    "fuel_tracker_database"
                )
                    .addMigrations(MIGRATION_1_2, MIGRATION_2_3, MIGRATION_3_4)
                    .build()
                INSTANCE = instance
                instance
            }
        }
    }
}
