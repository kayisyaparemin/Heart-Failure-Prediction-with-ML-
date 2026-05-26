package com.navisun.fueltracker.adapter

import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import com.navisun.fueltracker.data.TripEntry
import com.navisun.fueltracker.databinding.ItemTripBinding
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class TripAdapter(
    private val onItemClick: (TripEntry) -> Unit
) : ListAdapter<TripEntry, TripAdapter.TripViewHolder>(TripDiffCallback()) {

    private val dateFormat = SimpleDateFormat("dd.MM.yyyy HH:mm", Locale("tr", "TR"))

    inner class TripViewHolder(private val binding: ItemTripBinding) :
        RecyclerView.ViewHolder(binding.root) {

        fun bind(trip: TripEntry) {
            binding.tvTripDate.text = dateFormat.format(Date(trip.startTime))
            binding.tvTripDistance.text = String.format("%.1f km", trip.distanceKm)
            binding.tvTripDuration.text = "${trip.durationMinutes} dk"
            binding.tvTripAvgSpeed.text = String.format("%.0f km/s", trip.avgSpeedKmh)
            binding.tvTripMaxSpeed.text = String.format("%.0f km/s", trip.maxSpeedKmh)

            // Fuel type badge
            binding.tvTripFuelType.text = trip.fuelType
            val badgeColor = if (trip.fuelType == "LPG") {
                android.graphics.Color.parseColor("#00897b")
            } else {
                android.graphics.Color.parseColor("#f57c00")
            }
            val drawable = androidx.core.content.ContextCompat.getDrawable(
                binding.root.context,
                com.navisun.fueltracker.R.drawable.bg_fuel_badge
            )?.mutate()
            (drawable as? android.graphics.drawable.GradientDrawable)?.setColor(badgeColor)
            binding.tvTripFuelType.background = drawable

            binding.root.setOnClickListener {
                onItemClick(trip)
            }
        }
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): TripViewHolder {
        val binding = ItemTripBinding.inflate(LayoutInflater.from(parent.context), parent, false)
        return TripViewHolder(binding)
    }

    override fun onBindViewHolder(holder: TripViewHolder, position: Int) {
        holder.bind(getItem(position))
    }

    class TripDiffCallback : DiffUtil.ItemCallback<TripEntry>() {
        override fun areItemsTheSame(oldItem: TripEntry, newItem: TripEntry): Boolean =
            oldItem.id == newItem.id

        override fun areContentsTheSame(oldItem: TripEntry, newItem: TripEntry): Boolean =
            oldItem == newItem
    }
}
