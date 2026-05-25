package com.navisun.fueltracker.adapter

import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import com.navisun.fueltracker.data.FuelEntry
import com.navisun.fueltracker.databinding.ItemFuelEntryBinding
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class FuelHistoryAdapter(
    private val onDeleteClick: (FuelEntry) -> Unit
) : ListAdapter<FuelHistoryAdapter.FuelEntryWithConsumption, FuelHistoryAdapter.FuelEntryViewHolder>(
    FuelEntryDiffCallback()
) {

    data class FuelEntryWithConsumption(
        val entry: FuelEntry,
        val consumption: Double? = null // L/100km, null if not calculable
    )

    private val dateFormat = SimpleDateFormat("dd.MM.yyyy", Locale("tr", "TR"))

    inner class FuelEntryViewHolder(
        private val binding: ItemFuelEntryBinding
    ) : RecyclerView.ViewHolder(binding.root) {

        fun bind(item: FuelEntryWithConsumption) {
            val entry = item.entry

            // Tarih
            binding.tvDate.text = dateFormat.format(Date(entry.date))

            // Kilometre sayacı
            binding.tvOdometer.text = String.format("%,.0f km", entry.odometer)

            // Yakıt miktarı
            binding.tvFuelAmount.text = String.format("%.2f L", entry.fuelAmount)

            // Tüketim
            if (item.consumption != null) {
                binding.tvConsumption.text = String.format("%.1f L/100km", item.consumption)
            } else {
                binding.tvConsumption.text = "-- L/100km"
            }

            // Toplam maliyet
            val totalCost = entry.fuelAmount * entry.pricePerLiter
            binding.tvCost.text = String.format("%.2f ₺", totalCost)

            // Tam dolum göstergesi
            binding.tvFullTank.text = if (entry.fullTank) "Tam Dolum" else "Kısmi"

            // Litre fiyatı
            binding.tvPricePerLiter.text = String.format("%.2f ₺/L", entry.pricePerLiter)

            // Sil butonu
            binding.btnDelete.setOnClickListener {
                onDeleteClick(entry)
            }

            // Not varsa göster
            if (entry.note.isNotBlank()) {
                binding.tvNote.text = entry.note
                binding.tvNote.visibility = android.view.View.VISIBLE
            } else {
                binding.tvNote.visibility = android.view.View.GONE
            }
        }
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): FuelEntryViewHolder {
        val binding = ItemFuelEntryBinding.inflate(
            LayoutInflater.from(parent.context), parent, false
        )
        return FuelEntryViewHolder(binding)
    }

    override fun onBindViewHolder(holder: FuelEntryViewHolder, position: Int) {
        holder.bind(getItem(position))
    }

    class FuelEntryDiffCallback : DiffUtil.ItemCallback<FuelEntryWithConsumption>() {
        override fun areItemsTheSame(
            oldItem: FuelEntryWithConsumption,
            newItem: FuelEntryWithConsumption
        ): Boolean = oldItem.entry.id == newItem.entry.id

        override fun areContentsTheSame(
            oldItem: FuelEntryWithConsumption,
            newItem: FuelEntryWithConsumption
        ): Boolean = oldItem == newItem
    }

    /**
     * Girişleri ve hesaplanmış tüketimleri birleştirerek adapter'a gönderir.
     * Tüketim hesaplama: fullTank=true olan ardışık iki giriş arasında hesaplanır.
     */
    fun submitEntriesWithConsumption(entries: List<FuelEntry>) {
        val entriesAsc = entries.sortedBy { it.date }
        val result = mutableListOf<FuelEntryWithConsumption>()

        // Her entry için consumption hesapla
        val consumptionMap = mutableMapOf<Long, Double>()
        var previousFullTankEntry: FuelEntry? = null

        for (entry in entriesAsc) {
            if (entry.fullTank) {
                val prev = previousFullTankEntry
                if (prev != null) {
                    val kmDiff = entry.odometer - prev.odometer
                    if (kmDiff > 0) {
                        val consumption = (entry.fuelAmount / kmDiff) * 100.0
                        if (consumption in 1.0..50.0) {
                            consumptionMap[entry.id] = consumption
                        }
                    }
                }
                previousFullTankEntry = entry
            }
        }

        // En yeniden en eskiye (DESC) sırala
        val entriesDesc = entriesAsc.reversed()
        for (entry in entriesDesc) {
            result.add(FuelEntryWithConsumption(entry, consumptionMap[entry.id]))
        }

        submitList(result)
    }
}
