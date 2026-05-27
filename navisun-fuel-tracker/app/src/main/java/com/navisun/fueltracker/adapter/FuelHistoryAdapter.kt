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
        val tripKm: Double? = null,
        val consumption: Double? = null,  // L/100km
        val costPerKm: Double? = null     // TL/km
    )

    private val dateFormat = SimpleDateFormat("dd.MM.yyyy", Locale("tr", "TR"))

    inner class FuelEntryViewHolder(
        private val binding: ItemFuelEntryBinding
    ) : RecyclerView.ViewHolder(binding.root) {

        fun bind(item: FuelEntryWithConsumption) {
            val entry = item.entry

            binding.tvDate.text = dateFormat.format(Date(entry.date))

            binding.tvFuelAmount.text = String.format("%.2f L", entry.fuelAmount)

            binding.tvPricePerLiter.text = String.format("%.2f ₺/L", entry.pricePerLiter)

            val totalCost = entry.fuelAmount * entry.pricePerLiter
            binding.tvCost.text = String.format("%.2f ₺", totalCost)

            if (item.consumption != null) {
                binding.tvConsumption.text = String.format("%.1f L/100km", item.consumption)
            } else {
                binding.tvConsumption.text = "--"
            }

            if (item.costPerKm != null) {
                binding.tvCostPerKm.text = String.format("%.2f ₺/km", item.costPerKm)
            } else {
                binding.tvCostPerKm.text = "--"
            }

            if (item.tripKm != null) {
                binding.tvTripKm.text = String.format("%.1f km", item.tripKm)
            } else {
                binding.tvTripKm.text = "--"
            }

            binding.tvFullTank.visibility = if (entry.fullTank) android.view.View.VISIBLE else android.view.View.GONE

            val fuelTypeBadge = binding.tvFuelTypeBadge
            fuelTypeBadge.text = entry.fuelType
            if (entry.fuelType == "LPG") {
                fuelTypeBadge.setBackgroundColor(android.graphics.Color.parseColor("#00897b"))
            } else {
                fuelTypeBadge.setBackgroundColor(android.graphics.Color.parseColor("#f57c00"))
            }

            binding.btnDelete.setOnClickListener { onDeleteClick(entry) }

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

    fun submitEntriesWithConsumption(entries: List<FuelEntry>, gpsDistances: Map<Long, Double> = emptyMap()) {
        val entriesAsc = entries.sortedBy { it.date }

        // Odometer diff as fallback, grouped by fuel type
        val odometerDiffMap = mutableMapOf<Long, Double>()
        val byType = entriesAsc.groupBy { it.fuelType }
        for ((_, typeEntries) in byType) {
            val sorted = typeEntries.sortedBy { it.date }
            for (i in 1 until sorted.size) {
                val diff = sorted[i].odometer - sorted[i - 1].odometer
                if (diff > 0) odometerDiffMap[sorted[i].id] = diff
            }
        }

        val result = entriesAsc.reversed().map { entry ->
            val gpsKm = gpsDistances[entry.id]
            val km = if (gpsKm != null && gpsKm > 0.3) gpsKm else odometerDiffMap[entry.id]

            val consumption = if (km != null && km > 0) {
                val c = (entry.fuelAmount / km) * 100.0
                if (c in 0.5..100.0) c else null
            } else null

            val costPerKm = if (km != null && km > 0) {
                (entry.fuelAmount * entry.pricePerLiter) / km
            } else null

            FuelEntryWithConsumption(
                entry = entry,
                tripKm = km,
                consumption = consumption,
                costPerKm = costPerKm
            )
        }
        submitList(result)
    }
}
