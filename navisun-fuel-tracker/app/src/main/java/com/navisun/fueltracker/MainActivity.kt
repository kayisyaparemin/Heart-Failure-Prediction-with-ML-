package com.navisun.fueltracker

import android.Manifest
import android.app.AlertDialog
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.widget.EditText
import android.widget.Toast
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.localbroadcastmanager.content.LocalBroadcastManager
import com.navisun.fueltracker.databinding.ActivityMainBinding
import com.navisun.fueltracker.service.TripTrackingService
import com.navisun.fueltracker.viewmodel.FuelViewModel

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private val viewModel: FuelViewModel by viewModels()

    private val PREFS_NAME = "navisun_prefs"
    private val KEY_INITIAL_ODOMETER_SET = "initial_odometer_set"
    private val KEY_TRIP_RUNNING = "trip_running"
    private val LOCATION_PERMISSION_REQUEST = 1001

    private val speedReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            val speedKmh = intent?.getFloatExtra(TripTrackingService.EXTRA_SPEED_KMH, 0f) ?: 0f
            binding.tvSpeedValue.text = String.format("%.0f", speedKmh)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupButtons()
        observeViewModel()
        checkInitialOdometer()
        restoreTripButtonState()
    }

    override fun onResume() {
        super.onResume()
        viewModel.refreshStats()
        LocalBroadcastManager.getInstance(this).registerReceiver(
            speedReceiver,
            IntentFilter(TripTrackingService.ACTION_SPEED_UPDATE)
        )
        restoreTripButtonState()
    }

    override fun onPause() {
        super.onPause()
        LocalBroadcastManager.getInstance(this).unregisterReceiver(speedReceiver)
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

        binding.btnTripHistory.setOnClickListener {
            startActivity(Intent(this, TripHistoryActivity::class.java))
        }

        binding.btnTripToggle.setOnClickListener {
            val prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            val isRunning = prefs.getBoolean(KEY_TRIP_RUNNING, false)
            if (isRunning) {
                stopTrip()
            } else {
                startTrip()
            }
        }
    }

    private fun startTrip() {
        if (!hasLocationPermission()) {
            requestLocationPermission()
            return
        }
        val prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        prefs.edit().putBoolean(KEY_TRIP_RUNNING, true).apply()

        val intent = Intent(this, TripTrackingService::class.java).apply {
            action = TripTrackingService.ACTION_START
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(intent)
        } else {
            startService(intent)
        }

        updateTripButton(running = true)
        binding.tvSpeedValue.text = "0"
    }

    private fun stopTrip() {
        val prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        prefs.edit().putBoolean(KEY_TRIP_RUNNING, false).apply()

        val intent = Intent(this, TripTrackingService::class.java).apply {
            action = TripTrackingService.ACTION_STOP
        }
        startService(intent)

        updateTripButton(running = false)
        binding.tvSpeedValue.text = "0"
        Toast.makeText(this, getString(R.string.trip_saved), Toast.LENGTH_SHORT).show()
    }

    private fun updateTripButton(running: Boolean) {
        if (running) {
            binding.btnTripToggle.text = getString(R.string.stop_trip)
            binding.btnTripToggle.setBackgroundColor(
                ContextCompat.getColor(this, android.R.color.holo_red_dark)
            )
            binding.btnTripToggle.backgroundTintList =
                android.content.res.ColorStateList.valueOf(
                    ContextCompat.getColor(this, android.R.color.holo_red_dark)
                )
        } else {
            binding.btnTripToggle.text = getString(R.string.start_trip)
            binding.btnTripToggle.backgroundTintList =
                android.content.res.ColorStateList.valueOf(
                    ContextCompat.getColor(this, R.color.success)
                )
        }
    }

    private fun restoreTripButtonState() {
        val prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val isRunning = prefs.getBoolean(KEY_TRIP_RUNNING, false)
        updateTripButton(isRunning)
    }

    private fun checkInitialOdometer() {
        val prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val alreadySet = prefs.getBoolean(KEY_INITIAL_ODOMETER_SET, false)
        if (!alreadySet) {
            showInitialOdometerDialog()
        }
    }

    private fun showInitialOdometerDialog() {
        val editText = EditText(this).apply {
            inputType = android.text.InputType.TYPE_CLASS_NUMBER or
                    android.text.InputType.TYPE_NUMBER_FLAG_DECIMAL
            hint = getString(R.string.initial_odometer_hint)
            setTextColor(ContextCompat.getColor(context, R.color.text_primary))
            setHintTextColor(ContextCompat.getColor(context, R.color.text_secondary))
            setPadding(48, 24, 48, 24)
        }

        AlertDialog.Builder(this, R.style.NavisunDialogTheme)
            .setTitle(getString(R.string.initial_odometer_title))
            .setMessage(getString(R.string.initial_odometer_message))
            .setView(editText)
            .setCancelable(false)
            .setPositiveButton(getString(R.string.ok)) { _, _ ->
                val value = editText.text.toString().toDoubleOrNull()
                if (value != null && value > 0) {
                    val prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
                    prefs.edit()
                        .putBoolean(KEY_INITIAL_ODOMETER_SET, true)
                        .putFloat("initial_odometer_value", value.toFloat())
                        .apply()
                } else {
                    // If invalid, mark as set anyway to avoid infinite loop
                    val prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
                    prefs.edit().putBoolean(KEY_INITIAL_ODOMETER_SET, true).apply()
                }
            }
            .show()
    }

    private fun hasLocationPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            this, Manifest.permission.ACCESS_FINE_LOCATION
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun requestLocationPermission() {
        ActivityCompat.requestPermissions(
            this,
            arrayOf(
                Manifest.permission.ACCESS_FINE_LOCATION,
                Manifest.permission.ACCESS_COARSE_LOCATION
            ),
            LOCATION_PERMISSION_REQUEST
        )
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == LOCATION_PERMISSION_REQUEST) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startTrip()
            } else {
                Toast.makeText(
                    this,
                    getString(R.string.location_permission_required),
                    Toast.LENGTH_SHORT
                ).show()
            }
        }
    }

    private fun observeViewModel() {
        viewModel.stats.observe(this) { stats ->
            if (stats.lastConsumption != null) {
                binding.tvLastConsumptionValue.text =
                    String.format("%.1f", stats.lastConsumption)
            } else {
                binding.tvLastConsumptionValue.text = "--"
            }

            if (stats.averageConsumption != null) {
                binding.tvAvgConsumptionValue.text =
                    String.format("%.1f", stats.averageConsumption)
            } else {
                binding.tvAvgConsumptionValue.text = "--"
            }

            if (stats.totalCost > 0) {
                binding.tvTotalCostValue.text =
                    String.format("%.0f ₺", stats.totalCost)
            } else {
                binding.tvTotalCostValue.text = "--"
            }

            if (stats.totalKm > 0) {
                binding.tvTotalKmValue.text =
                    String.format("%.0f", stats.totalKm)
            } else {
                binding.tvTotalKmValue.text = "--"
            }
        }
    }
}
