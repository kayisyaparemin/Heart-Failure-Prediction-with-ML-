package com.navisun.fueltracker

import android.app.DatePickerDialog
import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.widget.Toast
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import com.navisun.fueltracker.data.FuelEntry
import com.navisun.fueltracker.databinding.ActivityAddFuelBinding
import com.navisun.fueltracker.viewmodel.FuelViewModel
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Locale

class AddFuelActivity : AppCompatActivity() {

    private lateinit var binding: ActivityAddFuelBinding
    private val viewModel: FuelViewModel by viewModels()

    private val calendar = Calendar.getInstance()
    private val dateFormat = SimpleDateFormat("dd.MM.yyyy", Locale("tr", "TR"))

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityAddFuelBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupToolbar()
        setupDatePicker()
        setupCostCalculation()
        setupSaveButton()

        // Başlangıçta bugünün tarihini göster
        updateDateDisplay()
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
            binding.tvCalculatedCost.text = String.format(
                getString(R.string.calculated_cost_format),
                totalCost
            )
            binding.tvCalculatedCost.visibility = android.view.View.VISIBLE
        } else {
            binding.tvCalculatedCost.visibility = android.view.View.INVISIBLE
        }
    }

    private fun setupSaveButton() {
        binding.btnSave.setOnClickListener {
            saveEntry()
        }
    }

    private fun saveEntry() {
        // Doğrulama
        val odometerStr = binding.etOdometer.text.toString().trim()
        val fuelAmountStr = binding.etFuelAmount.text.toString().trim()
        val pricePerLiterStr = binding.etPricePerLiter.text.toString().trim()

        if (odometerStr.isEmpty()) {
            binding.etOdometer.error = getString(R.string.error_required)
            binding.etOdometer.requestFocus()
            return
        }

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

        val odometer = odometerStr.toDoubleOrNull()
        if (odometer == null || odometer <= 0) {
            binding.etOdometer.error = getString(R.string.error_invalid_number)
            binding.etOdometer.requestFocus()
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
            odometer = odometer,
            fuelAmount = fuelAmount,
            pricePerLiter = pricePerLiter,
            fullTank = fullTank,
            note = note
        )

        viewModel.insert(entry)

        Toast.makeText(this, getString(R.string.entry_saved), Toast.LENGTH_SHORT).show()
        finish()
    }
}
