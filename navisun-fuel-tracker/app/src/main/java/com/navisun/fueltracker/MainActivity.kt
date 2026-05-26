package com.navisun.fueltracker

import android.Manifest
import android.app.AlertDialog
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.graphics.Color
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
    private val KEY_ACTIVE_FUEL_TYPE = "active_fuel_type"
    private val LOCATION_PERMISSION_REQUEST = 1001

    private val tripStateReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            val speedKmh = intent?.getFloatExtra(TripTrackingService.EXTRA_SPEED_KMH, 0f) ?: 0f
            val stateName = intent?.getStringExtra(TripTrackingService.EXTRA_STATE) ?: "IDLE"
            val distanceKm = intent?.getFloatExtra(TripTrackingService.EXTRA_DISTANCE_KM, 0f) ?: 0f

            binding.tvSpeedValue.text = String.format("%.0f", speedKmh)

            when (stateName) {
                "RECORDING" -> {
                    binding.tvTripStatus.text = String.format("⬤ Kaydediliyor • %.1f km", distanceKm)
                    binding.tvTripStatus.setTextColor(Color.parseColor("#e94560"))
                }
                "CONFIRMING_STOP" -> {
                    binding.tvTripStatus.text = getString(R.string.trip_status_stopping)
                    binding.tvTripStatus.setTextColor(Color.parseColor("#ff9800"))
                }
                else -> {
                    // IDLE or CONFIRMING_START
                    binding.tvTripStatus.text = getString(R.string.trip_status_idle)
                    binding.tvTripStatus.setTextColor(
                        ContextCompat.getColor(this@MainActivity, R.color.text_secondary)
                    )
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupButtons()
        observeViewModel()
        checkInitialOdometer()
        setupActiveFuelToggle()
        restoreActiveFuelState()

        // Request location permission if not granted
        if (!hasLocationPermission()) {
            requestLocationPermission()
        }

        // Start auto-detection service (always safe to call)
        val intent = Intent(this, TripTrackingService::class.java).apply {
            action = TripTrackingService.ACTION_START
        }
        ContextCompat.startForegroundService(this, intent)
    }

    override fun onResume() {
        super.onResume()
        viewModel.refreshStats()
        LocalBroadcastManager.getInstance(this).registerReceiver(
            tripStateReceiver,
            IntentFilter(TripTrackingService.ACTION_TRIP_STATE_UPDATE)
        )
        restoreActiveFuelState()
    }

    override fun onPause() {
        super.onPause()
        LocalBroadcastManager.getInstance(this).unregisterReceiver(tripStateReceiver)
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
            setTextColor(android.graphics.Color.parseColor("#212121"))
            setHintTextColor(android.graphics.Color.parseColor("#757575"))
            setBackgroundColor(android.graphics.Color.WHITE)
            setPadding(48, 24, 48, 24)
        }

        AlertDialog.Builder(this, R.style.NavisunDialogTheme)
            .setTitle(getString(R.string.initial_odometer_title))
            .setMessage(getString(R.string.initial_odometer_message))
            .setView(editText)
            .setCancelable(false)
            .setPositiveButton(getString(R.string.ok)) { _, _ ->
                val value = editText.text.toString().toDoubleOrNull()
                val prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
                if (value != null && value > 0) {
                    prefs.edit()
                        .putBoolean(KEY_INITIAL_ODOMETER_SET, true)
                        .putFloat("initial_odometer_value", value.toFloat())
                        .apply()
                } else {
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
                // Permission granted – service will start using GPS on next onStartCommand cycle
                val intent = Intent(this, TripTrackingService::class.java).apply {
                    action = TripTrackingService.ACTION_START
                }
                ContextCompat.startForegroundService(this, intent)
            } else {
                Toast.makeText(
                    this,
                    getString(R.string.location_permission_required),
                    Toast.LENGTH_SHORT
                ).show()
            }
        }
    }

    private fun setupActiveFuelToggle() {
        binding.btnFuelLpg.setOnClickListener {
            setActiveFuelType("LPG")
        }
        binding.btnFuelBenzin.setOnClickListener {
            setActiveFuelType("BENZİN")
        }
    }

    private fun setActiveFuelType(fuelType: String) {
        val prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        prefs.edit().putString(KEY_ACTIVE_FUEL_TYPE, fuelType).apply()
        updateActiveFuelUI(fuelType)
    }

    private fun restoreActiveFuelState() {
        val prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val activeFuel = prefs.getString(KEY_ACTIVE_FUEL_TYPE, "LPG") ?: "LPG"
        updateActiveFuelUI(activeFuel)
    }

    private fun updateActiveFuelUI(activeFuel: String) {
        val tealColor = android.graphics.Color.parseColor("#00897b")
        val amberColor = android.graphics.Color.parseColor("#f57c00")
        val cardBgColor = ContextCompat.getColor(this, R.color.bg_card)

        if (activeFuel == "LPG") {
            binding.btnFuelLpg.backgroundTintList =
                android.content.res.ColorStateList.valueOf(tealColor)
            binding.btnFuelBenzin.backgroundTintList =
                android.content.res.ColorStateList.valueOf(cardBgColor)
        } else {
            binding.btnFuelLpg.backgroundTintList =
                android.content.res.ColorStateList.valueOf(cardBgColor)
            binding.btnFuelBenzin.backgroundTintList =
                android.content.res.ColorStateList.valueOf(amberColor)
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
