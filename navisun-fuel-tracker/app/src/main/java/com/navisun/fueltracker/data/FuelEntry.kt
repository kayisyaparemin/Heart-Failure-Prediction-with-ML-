package com.navisun.fueltracker.data

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "fuel_entries")
data class FuelEntry(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val date: Long,            // Unix timestamp (milliseconds)
    val odometer: Double = 0.0, // Kilometre sayacı (artık kullanılmıyor, GPS'ten hesaplanıyor)
    val fuelAmount: Double,    // Doldurulan yakıt (litre)
    val pricePerLiter: Double, // Litre fiyatı (TL)
    val fullTank: Boolean,     // Tam dolum mu?
    val note: String = "",     // Opsiyonel not
    val fuelType: String = "BENZİN"  // Yakıt tipi: BENZİN veya LPG
)
