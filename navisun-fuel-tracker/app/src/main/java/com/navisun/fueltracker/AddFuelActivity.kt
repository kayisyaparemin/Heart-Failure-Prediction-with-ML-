package com.navisun.fueltracker

import android.app.DatePickerDialog
import android.content.res.ColorStateList
import android.graphics.Color
import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.view.View
import android.widget.Toast
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.navisun.fueltracker.data.FuelDatabase
import com.navisun.fueltracker.data.FuelEntry
import com.navisun.fueltracker.databinding.ActivityAddFuelBinding
import com.navisun.fueltracker.viewmodel.FuelViewModel
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Locale
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class AddFuelActivity : AppCompatActivity() {

    private lateinit var binding: ActivityAddFuelBinding
    private val viewModel: FuelViewModel by viewModels()

    private val calendar = Calendar.getInstance()
    private val dateFormat = SimpleDateFormat("dd.MM.yyyy", Locale("tr", "TR"))

    private var selectedFuelType = "LPG"
    private var gpsDistanceKm: Double = 0.0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityAddFuelBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Başlangıç yakıt tipini prefs'ten oku
        val prefs = getSharedPreferences("navisun_prefs", MODE_PRIVATE)
        selectedFuelType = prefs.getString("active_fuel_type", "LPG") ?: "LPG"

        setupToolbar()
        setupDatePicker()
        setupFuelTypeToggle()
        setupCostCalculation()
        setupSaveButton()

        updateDateDisplay()
        updateFuelTypeUI()
        loadGpsDistance()
    }

    private fun setupToolbar() {
        setSupportActionBar(binding.toolbar)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        supportActionBar?.title = getString(R.string.add_fuel)
        binding.toolbar.setNavigationOnClickListener { finish() }
    }

    private fun setupDatePicker() {
        binding.btnSelectDate.setOnClickListener {
            DatePickerDialog(
                this,
                { _, year, month, dayOfMonth ->
                    calendar.set(year, month, dayOfMonth)
                    updateDateDisplay()
                    loadGpsDistance()
                },
                calendar.get(Calendar.YEAR),
                calendar.get(Calendar.MONTH),
                calendar.get(Calendar.DAY_OF_MONTH)
            ).show()
        }
    }

    private fun updateDateDisplay() {
        binding.btnSelectDate.text = dateFormat.format(calendar.time)
    }

    private fun setupFuelTypeToggle() {
        binding.btnFuelBenzin.setOnClickListener {
            selectedFuelType = "BENZİN"
            updateFuelTypeUI()
            loadGpsDistance()
        }

        binding.btnFuelLpg.setOnClickListener {
            selectedFuelType = "LPG"
            updateFuelTypeUI()
            loadGpsDistance()
        }
    }

    private fun updateFuelTypeUI() {
        val lpgActive = ColorStateList.valueOf(Color.parseColor("#0288d1"))
        val benzinActive = ColorStateList.valueOf(Color.parseColor("#f57c00"))
        val inactive = ColorStateList.valueOf(Color.parseColor("#37474f"))

        if (selectedFuelType == "LPG") {
            binding.btnFuelLpg.backgroundTintList = lpgActive
            binding.btnFuelBenzin.backgroundTintList = inactive
        } else {
            binding.btnFuelBenzin.backgroundTintList = benzinActive
            binding.btnFuelLpg.backgroundTintList = inactive
        }
    }

    private fun loadGpsDistance() {
        val currentDate = calendar.timeInMillis
        val fuelType = selectedFuelType

        lifecycleScope.launch {
            val result = withContext(Dispatchers.IO) {
                val db = FuelDatabase.getDatabase(this@AddFuelActivity)
                val lastEntry = db.fuelDao().getLastEntryByType(fuelType)
                val fromTime = lastEntry?.date ?: 0L
                // getTripsBetweenAll + segment km için doğru yakıt tipini kullan
                val trips = db.tripDao().getTripsBetweenAll(fromTime, currentDate)
                trips.sumOf { it.getKmForFuelType(fuelType) }
            }

            gpsDistanceKm = result
            if (gpsDistanceKm > 0.5) {
                binding.tvGpsDistance.text = String.format("%.1f km", gpsDistanceKm)
                binding.cardGpsDistance.visibility = View.VISIBLE
            } else {
                binding.cardGpsDistance.visibility = View.GONE
            }
            calculateAndDisplayCost()
        }
    }

    private fun setupCostCalculation() {
        val watcher = object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {}
            override fun afterTextChanged(s: Editable?) {
                calculateAndDisplayCost()
            }
        }
        binding.etFuelAmount.addTextChangedListener(watcher)
        binding.etPricePerLiter.addTextChangedListener(watcher)
    }

    private fun calculateAndDisplayCost() {
        val fuelAmount = binding.etFuelAmount.text.toString().toDoubleOrNull()
        val pricePerLiter = binding.etPricePerLiter.text.toString().toDoubleOrNull()

        if (fuelAmount != null && pricePerLiter != null && fuelAmount > 0 && pricePerLiter > 0) {
            val totalCost = fuelAmount * pricePerLiter
            binding.tvCalculatedCost.text = String.format("%.2f ₺", totalCost)
            binding.cardCalculatedCost.visibility = View.VISIBLE

            if (gpsDistanceKm > 0.5) {
                val costPerKm = totalCost / gpsDistanceKm
                binding.tvCostPerKm.text = String.format("%.2f ₺/km", costPerKm)
                binding.cardCostPerKm.visibility = View.VISIBLE
            } else {
                binding.cardCostPerKm.visibility = View.GONE
            }
        } else {
            binding.cardCalculatedCost.visibility = View.INVISIBLE
            binding.cardCostPerKm.visibility = View.GONE
        }
    }

    private fun setupSaveButton() {
        binding.btnSave.setOnClickListener { saveEntry() }
    }

    private fun saveEntry() {
        val fuelAmountStr = binding.etFuelAmount.text.toString().trim()
        val pricePerLiterStr = binding.etPricePerLiter.text.toString().trim()

        if (fuelAmountStr.isEmpty()) {
            binding.etFuelAmount.error = getString(R.string.error_required)
            binding.etFuelAmount.requestFocus()
            return
        }

        if (pricePerLiterStr.isEmpty()) {
            binding.etPricePerLiter.error = getString(R.string.error_required)
            binding.etPricePerLiter.requestFocus()
            return
        }

        val fuelAmount = fuelAmountStr.toDoubleOrNull()
        if (fuelAmount == null || fuelAmount <= 0) {
            binding.etFuelAmount.error = getString(R.string.error_invalid_number)
            binding.etFuelAmount.requestFocus()
            return
        }

        val pricePerLiter = pricePerLiterStr.toDoubleOrNull()
        if (pricePerLiter == null || pricePerLiter <= 0) {
            binding.etPricePerLiter.error = getString(R.string.error_invalid_number)
            binding.etPricePerLiter.requestFocus()
            return
        }

        val fullTank = binding.cbFullTank.isChecked
        val note = binding.etNote.text.toString().trim()

        val entry = FuelEntry(
            date = calendar.timeInMillis,
            odometer = 0.0,
            fuelAmount = fuelAmount,
            pricePerLiter = pricePerLiter,
            fullTank = fullTank,
            note = note,
            fuelType = selectedFuelType
        )

        viewModel.insert(entry)
        Toast.makeText(this, getString(R.string.entry_saved), Toast.LENGTH_SHORT).show()
        finish()
    }
}
