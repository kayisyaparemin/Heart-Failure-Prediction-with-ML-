package com.navisun.fueltracker

import android.content.res.ColorStateList
import android.graphics.Color
import android.os.Bundle
import android.view.View
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import com.navisun.fueltracker.databinding.ActivityStatsBinding
import com.navisun.fueltracker.data.FuelEntry
import com.navisun.fueltracker.viewmodel.FuelStats
import com.navisun.fueltracker.viewmodel.FuelTypeStats
import com.navisun.fueltracker.viewmodel.FuelViewModel
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class StatsActivity : AppCompatActivity() {

    private lateinit var binding: ActivityStatsBinding
    private val viewModel: FuelViewModel by viewModels()
    private val dateFormat = SimpleDateFormat("dd.MM.yyyy", Locale("tr", "TR"))

    private var currentFilter = "ALL"
    private var latestStats: FuelStats? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityStatsBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupToolbar()
        setupFilterButtons()
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

    private fun setupFilterButtons() {
        binding.btnFilterAll.setOnClickListener { applyFilter("ALL") }
        binding.btnFilterLpg.setOnClickListener { applyFilter("LPG") }
        binding.btnFilterBenzin.setOnClickListener { applyFilter("BENZİN") }
        updateFilterUI("ALL")
    }

    private fun applyFilter(filter: String) {
        currentFilter = filter
        updateFilterUI(filter)
        latestStats?.let { renderStats(it) }
    }

    private fun updateFilterUI(filter: String) {
        val allActive    = ColorStateList.valueOf(Color.parseColor("#e94560"))
        val lpgActive    = ColorStateList.valueOf(Color.parseColor("#0288d1"))
        val benzinActive = ColorStateList.valueOf(Color.parseColor("#f57c00"))
        val inactive     = ColorStateList.valueOf(Color.parseColor("#37474f"))

        binding.btnFilterAll.backgroundTintList    = if (filter == "ALL")     allActive    else inactive
        binding.btnFilterLpg.backgroundTintList    = if (filter == "LPG")     lpgActive    else inactive
        binding.btnFilterBenzin.backgroundTintList = if (filter == "BENZİN")  benzinActive else inactive
    }

    private fun observeViewModel() {
        viewModel.stats.observe(this) { stats ->
            latestStats = stats
            renderStats(stats)
        }
    }

    private fun renderStats(stats: FuelStats) {
        when (currentFilter) {
            "LPG"    -> renderTypeStats(stats.lpgStats)
            "BENZİN" -> renderTypeStats(stats.benzinStats)
            else     -> renderAllStats(stats)
        }
    }

    private fun renderAllStats(stats: FuelStats) {
        binding.tvTotalEntriesValue.text = stats.totalEntries.toString()
        binding.tvTotalKmValue.text = if (stats.totalKm > 0) String.format("%.0f", stats.totalKm) else "--"
        binding.tvAvgConsumptionValue.text = stats.averageConsumption?.let { String.format("%.2f L/100km", it) } ?: "--"
        binding.tvBestConsumptionValue.text = stats.bestConsumption?.let { String.format("%.2f L/100km", it) } ?: "--"
        binding.tvWorstConsumptionValue.text = stats.worstConsumption?.let { String.format("%.2f L/100km", it) } ?: "--"
        binding.tvTotalCostValue.text = if (stats.totalCost > 0) String.format("%.2f ₺", stats.totalCost) else "--"
        binding.tvTotalFuelValue.text = if (stats.totalFuel > 0) String.format("%.2f L", stats.totalFuel) else "--"
        binding.tvAvgPriceValue.text = stats.avgPricePerLiter?.let { String.format("%.2f ₺/L", it) } ?: "--"
        binding.tvAvgCostPerKm.text = stats.avgCostPerKm?.let { String.format("%.2f ₺/km", it) } ?: "--"

        binding.tvFuelTypeSection.visibility = View.VISIBLE
        binding.cardLpgStats.visibility = View.VISIBLE
        binding.cardBenzinStats.visibility = View.VISIBLE

        // LPG
        val lpg = stats.lpgStats
        binding.tvLpgAvgConsumption.text = lpg.averageConsumption?.let { String.format("%.2f L/100km", it) } ?: "--"
        binding.tvLpgTotalCost.text = if (lpg.totalCost > 0) String.format("%.2f ₺", lpg.totalCost) else "--"
        binding.tvLpgTotalFuel.text = if (lpg.totalFuel > 0) String.format("%.2f L", lpg.totalFuel) else "--"

        // BENZİN
        val benzin = stats.benzinStats
        binding.tvBenzinAvgConsumption.text = benzin.averageConsumption?.let { String.format("%.2f L/100km", it) } ?: "--"
        binding.tvBenzinTotalCost.text = if (benzin.totalCost > 0) String.format("%.2f ₺", benzin.totalCost) else "--"
        binding.tvBenzinTotalFuel.text = if (benzin.totalFuel > 0) String.format("%.2f L", benzin.totalFuel) else "--"

        updateRecentConsumptions(stats.recentConsumptions)
    }

    private fun renderTypeStats(typeStats: FuelTypeStats) {
        binding.tvTotalEntriesValue.text = typeStats.totalEntries.toString()
        binding.tvTotalKmValue.text = if (typeStats.totalKm > 0) String.format("%.0f", typeStats.totalKm) else "--"
        binding.tvAvgConsumptionValue.text = typeStats.averageConsumption?.let { String.format("%.2f L/100km", it) } ?: "--"
        binding.tvBestConsumptionValue.text = typeStats.bestConsumption?.let { String.format("%.2f L/100km", it) } ?: "--"
        binding.tvWorstConsumptionValue.text = typeStats.worstConsumption?.let { String.format("%.2f L/100km", it) } ?: "--"
        binding.tvTotalCostValue.text = if (typeStats.totalCost > 0) String.format("%.2f ₺", typeStats.totalCost) else "--"
        binding.tvTotalFuelValue.text = if (typeStats.totalFuel > 0) String.format("%.2f L", typeStats.totalFuel) else "--"
        binding.tvAvgPriceValue.text = typeStats.avgPricePerLiter?.let { String.format("%.2f ₺/L", it) } ?: "--"
        binding.tvAvgCostPerKm.text = typeStats.avgCostPerKm?.let { String.format("%.2f ₺/km", it) } ?: "--"

        binding.tvFuelTypeSection.visibility = View.GONE
        binding.cardLpgStats.visibility = View.GONE
        binding.cardBenzinStats.visibility = View.GONE

        updateRecentConsumptions(typeStats.recentConsumptions)
    }

    private fun updateRecentConsumptions(consumptions: List<Pair<FuelEntry, Double>>) {
        binding.containerRecentConsumptions.removeAllViews()

        if (consumptions.isEmpty()) {
            binding.tvNoRecentData.visibility = View.VISIBLE
            return
        }
        binding.tvNoRecentData.visibility = View.GONE

        for ((entry, consumption) in consumptions) {
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
            tvKm.text = String.format("[%s]", entry.fuelType)

            binding.containerRecentConsumptions.addView(itemView)
        }
    }
}
