package com.navisun.fueltracker

import android.content.Intent
import android.os.Bundle
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import com.navisun.fueltracker.databinding.ActivityMainBinding
import com.navisun.fueltracker.viewmodel.FuelViewModel

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private val viewModel: FuelViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupButtons()
        observeViewModel()
    }

    override fun onResume() {
        super.onResume()
        // İstatistikleri yenile her geri dönüşte
        viewModel.refreshStats()
    }

    private fun setupButtons() {
        binding.btnAddFuel.setOnClickListener {
            startActivity(Intent(this, AddFuelActivity::class.java))
        }

        binding.btnHistory.setOnClickListener {
            startActivity(Intent(this, HistoryActivity::class.java))
        }

        binding.btnStats.setOnClickListener {
            startActivity(Intent(this, StatsActivity::class.java))
        }
    }

    private fun observeViewModel() {
        viewModel.stats.observe(this) { stats ->
            // Son tüketim
            if (stats.lastConsumption != null) {
                binding.tvLastConsumptionValue.text =
                    String.format("%.1f", stats.lastConsumption)
            } else {
                binding.tvLastConsumptionValue.text = "--"
            }

            // Ortalama tüketim
            if (stats.averageConsumption != null) {
                binding.tvAvgConsumptionValue.text =
                    String.format("%.1f", stats.averageConsumption)
            } else {
                binding.tvAvgConsumptionValue.text = "--"
            }

            // Toplam maliyet
            if (stats.totalCost > 0) {
                binding.tvTotalCostValue.text =
                    String.format("%.0f ₺", stats.totalCost)
            } else {
                binding.tvTotalCostValue.text = "--"
            }

            // Toplam km
            if (stats.totalKm > 0) {
                binding.tvTotalKmValue.text =
                    String.format("%.0f", stats.totalKm)
            } else {
                binding.tvTotalKmValue.text = "--"
            }
        }
    }
}
