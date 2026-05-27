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

class FuelHistoryAdapter : ListAdapter<FuelEntry, FuelHistoryAdapter.FuelEntryViewHolder>(
    FuelEntryDiffCallback()
) {

    private val dateFormat = SimpleDateFormat("dd.MM.yyyy", Locale("tr", "TR"))

    inner class FuelEntryViewHolder(
        private val binding: ItemFuelEntryBinding
    ) : RecyclerView.ViewHolder(binding.root) {

        fun bind(entry: FuelEntry) {
            binding.tvDate.text = dateFormat.format(Date(entry.date))
            binding.tvFuelAmount.text = String.format("%.2f L", entry.fuelAmount)
            binding.tvPricePerLiter.text = String.format("%.2f ₺/L", entry.pricePerLiter)
            binding.tvCost.text = String.format("%.2f ₺", entry.fuelAmount * entry.pricePerLiter)
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

    class FuelEntryDiffCallback : DiffUtil.ItemCallback<FuelEntry>() {
        override fun areItemsTheSame(oldItem: FuelEntry, newItem: FuelEntry): Boolean =
            oldItem.id == newItem.id

        override fun areContentsTheSame(oldItem: FuelEntry, newItem: FuelEntry): Boolean =
            oldItem == newItem
    }

    fun submitEntries(entries: List<FuelEntry>) {
        submitList(entries.sortedByDescending { it.date })
    }
}
