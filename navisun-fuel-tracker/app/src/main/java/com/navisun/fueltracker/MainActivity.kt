package com.navisun.fueltracker

import android.Manifest
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.graphics.Color
import android.os.Bundle
import android.view.Menu
import android.view.MenuItem
import android.widget.Toast
import androidx.activity.viewModels
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import androidx.localbroadcastmanager.content.LocalBroadcastManager
import com.navisun.fueltracker.databinding.ActivityMainBinding
import com.navisun.fueltracker.service.TripTrackingService
import com.navisun.fueltracker.viewmodel.FuelViewModel
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private val viewModel: FuelViewModel by viewModels()

    private val PREFS_NAME = "navisun_prefs"
    private val KEY_ACTIVE_FUEL_TYPE = "active_fuel_type"
    private val LOCATION_PERMISSION_REQUEST = 1001

    private var activeFuelType: String = "LPG"
    private var statsFilter: String = "ALL"
    private var latestStats: com.navisun.fueltracker.viewmodel.FuelStats? = null

    private val tripStateReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            val speedKmh = intent?.getFloatExtra(TripTrackingService.EXTRA_SPEED_KMH, 0f) ?: 0f
            val stateName = intent?.getStringExtra(TripTrackingService.EXTRA_STATE) ?: "IDLE"
            val distanceKm = intent?.getFloatExtra(TripTrackingService.EXTRA_DISTANCE_KM, 0f) ?: 0f

            binding.tvSpeed.text = String.format("%.0f", speedKmh)

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

        setupToolbar()
        selectTab("DRIVE")

        val prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        activeFuelType = prefs.getString(KEY_ACTIVE_FUEL_TYPE, "LPG") ?: "LPG"
        updateActiveFuelUI()

        setupTabListeners()
        setupFuelFilterButtons()
        setupNavigationListeners()
        observeViewModel()

        if (!hasLocationPermission()) {
            requestLocationPermission()
        }

        val serviceIntent = Intent(this, TripTrackingService::class.java).apply {
            action = TripTrackingService.ACTION_START
        }
        ContextCompat.startForegroundService(this, serviceIntent)
    }

    override fun onResume() {
        super.onResume()
        viewModel.refreshStats()
        LocalBroadcastManager.getInstance(this).registerReceiver(
            tripStateReceiver,
            IntentFilter(TripTrackingService.ACTION_TRIP_STATE_UPDATE)
        )
        val prefs = getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        activeFuelType = prefs.getString(KEY_ACTIVE_FUEL_TYPE, "LPG") ?: "LPG"
        updateActiveFuelUI()
    }

    override fun onPause() {
        super.onPause()
        LocalBroadcastManager.getInstance(this).unregisterReceiver(tripStateReceiver)
    }

    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        menuInflater.inflate(R.menu.menu_main, menu)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return when (item.itemId) {
            R.id.action_simulate_trip -> {
                val serviceIntent = Intent(this, TripTrackingService::class.java).apply {
                    action = TripTrackingService.ACTION_SIMULATE_TRIP
                }
                ContextCompat.startForegroundService(this, serviceIntent)
                Toast.makeText(this,
                    "Simülasyon başladı! ~85 saniye sonra sürüş kaydedilecek.",
                    Toast.LENGTH_LONG).show()
                true
            }
            R.id.action_insert_test_data -> {
                AlertDialog.Builder(this)
                    .setTitle("Test Verisi Ekle")
                    .setMessage("5 sürüş ve 5 yakıt kaydı eklenecek (İstanbul güzergahları, LPG + BENZİN). Devam edilsin mi?")
                    .setPositiveButton("Ekle") { _, _ ->
                        lifecycleScope.launch {
                            TestDataHelper.insertTestData(applicationContext)
                            viewModel.refreshStats()
                            Toast.makeText(this@MainActivity, "Test verisi eklendi ✓", Toast.LENGTH_SHORT).show()
                        }
                    }
                    .setNegativeButton("İptal", null)
                    .show()
                true
            }
            R.id.action_clear_all_data -> {
                AlertDialog.Builder(this)
                    .setTitle("Tüm Veriyi Sil")
                    .setMessage("Tüm sürüş ve yakıt kayıtları silinecek. Bu işlem geri alınamaz!")
                    .setPositiveButton("Sil") { _, _ ->
                        lifecycleScope.launch {
                            TestDataHelper.clearAllData(applicationContext)
                            viewModel.refreshStats()
                            Toast.makeText(this@MainActivity, "Tüm veri silindi.", Toast.LENGTH_SHORT).show()
                        }
                    }
                    .setNegativeButton("İptal", null)
                    .show()
                true
            }
            else -> super.onOptionsItemSelected(item)
        }
    }

    private fun setupToolbar() {
        setSupportActionBar(binding.toolbar)
        supportActionBar?.setDisplayHomeAsUpEnabled(false)
    }

    private fun selectTab(tab: String) {
        val driveActiveColor  = ColorStateList.valueOf(Color.parseColor("#4caf50"))
        val fuelActiveColor   = ColorStateList.valueOf(Color.parseColor("#e94560"))
        val inactiveColor     = ColorStateList.valueOf(Color.parseColor("#37474f"))
        if (tab == "DRIVE") {
            binding.viewFlipper.displayedChild = 0
            binding.btnTabDrive.backgroundTintList = driveActiveColor
            binding.btnTabFuel.backgroundTintList  = inactiveColor
        } else {
            binding.viewFlipper.displayedChild = 1
            binding.btnTabFuel.backgroundTintList  = fuelActiveColor
            binding.btnTabDrive.backgroundTintList = inactiveColor
        }
    }

    private fun setupTabListeners() {
        binding.btnTabDrive.setOnClickListener { selectTab("DRIVE") }
        binding.btnTabFuel.setOnClickListener { selectTab("FUEL") }
    }

    private fun setupFuelFilterButtons() {
        binding.btnFuelFilterAll.setOnClickListener    { applyStatsFilter("ALL") }
        binding.btnFuelFilterLpg.setOnClickListener    { applyStatsFilter("LPG") }
        binding.btnFuelFilterBenzin.setOnClickListener { applyStatsFilter("BENZİN") }
        updateStatsFilterUI("ALL")
    }

    private fun applyStatsFilter(filter: String) {
        statsFilter = filter
        updateStatsFilterUI(filter)
        latestStats?.let { renderMainStats(it) }
    }

    private fun updateStatsFilterUI(filter: String) {
        val allActive    = ColorStateList.valueOf(Color.parseColor("#e94560"))
        val lpgActive    = ColorStateList.valueOf(Color.parseColor("#0288d1"))
        val benzinActive = ColorStateList.valueOf(Color.parseColor("#f57c00"))
        val inactive     = ColorStateList.valueOf(Color.parseColor("#37474f"))
        binding.btnFuelFilterAll.backgroundTintList    = if (filter == "ALL")     allActive    else inactive
        binding.btnFuelFilterLpg.backgroundTintList    = if (filter == "LPG")     lpgActive    else inactive
        binding.btnFuelFilterBenzin.backgroundTintList = if (filter == "BENZİN")  benzinActive else inactive
    }

    private fun renderMainStats(stats: com.navisun.fueltracker.viewmodel.FuelStats) {
        val lastCostPerKm: Double?
        val avgCostPerKm: Double?
        val totalCost: Double
        val totalKm: Double

        when (statsFilter) {
            "LPG" -> {
                val s = stats.lpgStats
                lastCostPerKm = s.lastCostPerKm
                avgCostPerKm  = s.avgCostPerKm
                totalCost     = s.totalCost
                totalKm       = s.totalKm
            }
            "BENZİN" -> {
                val s = stats.benzinStats
                lastCostPerKm = s.lastCostPerKm
                avgCostPerKm  = s.avgCostPerKm
                totalCost     = s.totalCost
                totalKm       = s.totalKm
            }
            else -> {
                lastCostPerKm = stats.lastCostPerKm
                avgCostPerKm  = stats.avgCostPerKm
                totalCost     = stats.totalCost
                totalKm       = stats.totalKm
            }
        }

        binding.tvLastConsumption.text = lastCostPerKm?.let { String.format("%.2f", it) } ?: "--"
        binding.tvAvgConsumption.text  = avgCostPerKm?.let  { String.format("%.2f", it) } ?: "--"
        binding.tvTotalCost.text       = if (totalCost > 0) String.format("%.0f", totalCost) else "--"
        binding.tvTotalKm.text         = if (totalKm > 0)   String.format("%.0f", totalKm)  else "--"
    }

    private fun setupNavigationListeners() {
        binding.btnAddFuel.setOnClickListener {
            startActivity(Intent(this, AddFuelActivity::class.java))
        }

        binding.btnFuelHistory.setOnClickListener {
            startActivity(Intent(this, HistoryActivity::class.java))
        }

        binding.btnStats.setOnClickListener {
            startActivity(Intent(this, StatsActivity::class.java))
        }

        binding.btnTripHistory.setOnClickListener {
            startActivity(Intent(this, TripHistoryActivity::class.java))
        }

        binding.btnLpg.setOnClickListener {
            activeFuelType = "LPG"
            getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
                .edit().putString(KEY_ACTIVE_FUEL_TYPE, "LPG").apply()
            updateActiveFuelUI()
            broadcastFuelTypeChange("LPG")
        }

        binding.btnBenzin.setOnClickListener {
            activeFuelType = "BENZİN"
            getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
                .edit().putString(KEY_ACTIVE_FUEL_TYPE, "BENZİN").apply()
            updateActiveFuelUI()
            broadcastFuelTypeChange("BENZİN")
        }
    }

    private fun broadcastFuelTypeChange(fuelType: String) {
        LocalBroadcastManager.getInstance(this).sendBroadcast(
            Intent(TripTrackingService.ACTION_FUEL_TYPE_CHANGED)
                .putExtra(TripTrackingService.EXTRA_FUEL_TYPE, fuelType)
        )
    }

    private fun updateActiveFuelUI() {
        val lpgActive = ColorStateList.valueOf(Color.parseColor("#0288d1"))
        val lpgInactive = ColorStateList.valueOf(Color.parseColor("#37474f"))
        val benzinActive = ColorStateList.valueOf(Color.parseColor("#f57c00"))
        val benzinInactive = ColorStateList.valueOf(Color.parseColor("#37474f"))
        if (activeFuelType == "LPG") {
            binding.btnLpg.backgroundTintList = lpgActive
            binding.btnBenzin.backgroundTintList = benzinInactive
        } else {
            binding.btnLpg.backgroundTintList = lpgInactive
            binding.btnBenzin.backgroundTintList = benzinActive
        }
    }

    private fun observeViewModel() {
        viewModel.stats.observe(this) { stats ->
            latestStats = stats
            renderMainStats(stats)
        }
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
                val intent = Intent(this, TripTrackingService::class.java).apply {
                    action = TripTrackingService.ACTION_START
                }
                ContextCompat.startForegroundService(this, intent)
            } else {
                Toast.makeText(this, getString(R.string.location_permission_required), Toast.LENGTH_SHORT).show()
            }
        }
    }
}
