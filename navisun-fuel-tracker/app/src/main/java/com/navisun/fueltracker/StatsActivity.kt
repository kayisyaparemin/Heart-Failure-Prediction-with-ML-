package com.navisun.fueltracker

import android.os.Bundle
import android.view.View
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import com.navisun.fueltracker.databinding.ActivityStatsBinding
import com.navisun.fueltracker.viewmodel.FuelStats
import com.navisun.fueltracker.viewmodel.FuelViewModel
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class StatsActivity : AppCompatActivity() {

    private lateinit var binding: ActivityStatsBinding
    private val viewModel: FuelViewModel by viewModels()
    private val dateFormat = SimpleDateFormat("dd.MM.yyyy", Locale("tr", "TR"))

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityStatsBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupToolbar()
        observeViewModel()
    }

    override fun onResume() {
        super.onResume()
        viewModel.refreshStats()
    }

    private fun setupToolbar() {
        setSupportActionBar(binding.toolbar)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        supportActionBar?.title = getString(R.string.stats)
        binding.toolbar.setNavigationOnClickListener { finish() }
    }

    private fun observeViewModel() {
        viewModel.stats.observe(this) { stats ->
            // Toplam kayıt sayısı
            binding.tvTotalEntriesValue.text = stats.totalEntries.toString()

            // Ortalama tüketim
            binding.tvAvgConsumptionValue.text = if (stats.averageConsumption != null) {
                String.format("%.2f L/100km", stats.averageConsumption)
            } else "--"

            // En iyi tüketim (en düşük L/100km)
            binding.tvBestConsumptionValue.text = if (stats.bestConsumption != null) {
                String.format("%.2f L/100km", stats.bestConsumption)
            } else "--"

            // En kötü tüketim (en yüksek L/100km)
            binding.tvWorstConsumptionValue.text = if (stats.worstConsumption != null) {
                String.format("%.2f L/100km", stats.worstConsumption)
            } else "--"

            // Toplam yakıt harcaması
            binding.tvTotalCostValue.text = if (stats.totalCost > 0) {
                String.format("%.2f ₺", stats.totalCost)
            } else "--"

            // Toplam doldurulan yakıt
            binding.tvTotalFuelValue.text = if (stats.totalFuel > 0) {
                String.format("%.2f L", stats.totalFuel)
            } else "--"

            // Ortalama litre fiyatı
            binding.tvAvgPriceValue.text = if (stats.avgPricePerLiter != null) {
                String.format("%.2f ₺/L", stats.avgPricePerLiter)
            } else "--"

            // Toplam km
            binding.tvTotalKmValue.text = if (stats.totalKm > 0) {
                String.format("%.0f km", stats.totalKm)
            } else "--"

            // Ortalama TL/km
            binding.tvAvgCostPerKm.text = if (stats.avgCostPerKm != null) {
                String.format("%.2f ₺/km", stats.avgCostPerKm)
            } else "--"

            // LPG Tankı section
            val lpg = stats.lpgStats
            if (lpg.totalEntries > 0) {
                binding.tvLpgAvgConsumption.text = if (lpg.averageConsumption != null) {
                    String.format("%.2f L/100km", lpg.averageConsumption)
                } else "--"
                binding.tvLpgTotalCost.text = String.format("%.2f ₺", lpg.totalCost)
                binding.tvLpgTotalFuel.text = String.format("%.2f L", lpg.totalFuel)
            } else {
                binding.tvLpgAvgConsumption.text = "--"
                binding.tvLpgTotalCost.text = "--"
                binding.tvLpgTotalFuel.text = "--"
            }

            // Benzin Tankı section
            val benzin = stats.benzinStats
            if (benzin.totalEntries > 0) {
                binding.tvBenzinAvgConsumption.text = if (benzin.averageConsumption != null) {
                    String.format("%.2f L/100km", benzin.averageConsumption)
                } else "--"
                binding.tvBenzinTotalCost.text = String.format("%.2f ₺", benzin.totalCost)
                binding.tvBenzinTotalFuel.text = String.format("%.2f L", benzin.totalFuel)
            } else {
                binding.tvBenzinAvgConsumption.text = "--"
                binding.tvBenzinTotalCost.text = "--"
                binding.tvBenzinTotalFuel.text = "--"
            }

            // Son 5 dolum tüketimleri
            updateRecentConsumptions(stats)
        }
    }

    private fun updateRecentConsumptions(stats: FuelStats) {
        binding.containerRecentConsumptions.removeAllViews()

        if (stats.recentConsumptions.isEmpty()) {
            binding.tvNoRecentData.visibility = View.VISIBLE
            return
        }

        binding.tvNoRecentData.visibility = View.GONE

        for ((entry, consumption) in stats.recentConsumptions) {
            val itemView = layoutInflater.inflate(
                R.layout.item_recent_consumption,
                binding.containerRecentConsumptions,
                false
            )

            val tvDate = itemView.findViewById<android.widget.TextView>(R.id.tv_recent_date)
            val tvConsumption = itemView.findViewById<android.widget.TextView>(R.id.tv_recent_consumption)
            val tvKm = itemView.findViewById<android.widget.TextView>(R.id.tv_recent_km)

            tvDate.text = dateFormat.format(Date(entry.date))
            tvConsumption.text = String.format("%.1f L/100km", consumption)
            tvKm.text = String.format("%.0f km  [%s]", entry.odometer, entry.fuelType)

            binding.containerRecentConsumptions.addView(itemView)
        }
    }
}
