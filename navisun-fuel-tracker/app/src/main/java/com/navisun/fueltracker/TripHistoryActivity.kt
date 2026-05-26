package com.navisun.fueltracker

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.View
import androidx.activity.viewModels
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.ItemTouchHelper
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.navisun.fueltracker.adapter.TripAdapter
import com.navisun.fueltracker.data.TripEntry
import com.navisun.fueltracker.databinding.ActivityTripHistoryBinding
import com.navisun.fueltracker.viewmodel.TripViewModel

class TripHistoryActivity : AppCompatActivity() {

    private lateinit var binding: ActivityTripHistoryBinding
    private val viewModel: TripViewModel by viewModels()
    private lateinit var adapter: TripAdapter

    private val LOCATION_PERMISSION_REQUEST = 2001

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityTripHistoryBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupToolbar()
        setupRecyclerView()
        observeViewModel()
        checkLocationPermission()
    }

    private fun setupToolbar() {
        setSupportActionBar(binding.toolbar)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        supportActionBar?.title = getString(R.string.trip_history)
        binding.toolbar.setNavigationOnClickListener { finish() }
    }

    private fun setupRecyclerView() {
        adapter = TripAdapter { trip ->
            openTripDetail(trip)
        }

        binding.recyclerTrips.layoutManager = LinearLayoutManager(this)
        binding.recyclerTrips.adapter = adapter

        // Swipe to delete
        val itemTouchHelper = ItemTouchHelper(object : ItemTouchHelper.SimpleCallback(
            0, ItemTouchHelper.LEFT or ItemTouchHelper.RIGHT
        ) {
            override fun onMove(
                recyclerView: RecyclerView,
                viewHolder: RecyclerView.ViewHolder,
                target: RecyclerView.ViewHolder
            ): Boolean = false

            override fun onSwiped(viewHolder: RecyclerView.ViewHolder, direction: Int) {
                val position = viewHolder.adapterPosition
                val trip = adapter.currentList[position]
                showDeleteConfirmation(trip)
                // Restore item visually until confirmed
                adapter.notifyItemChanged(position)
            }
        })
        itemTouchHelper.attachToRecyclerView(binding.recyclerTrips)
    }

    private fun observeViewModel() {
        viewModel.allTrips.observe(this) { trips ->
            adapter.submitList(trips)
            if (trips.isEmpty()) {
                binding.layoutNoTrips.visibility = View.VISIBLE
                binding.recyclerTrips.visibility = View.GONE
            } else {
                binding.layoutNoTrips.visibility = View.GONE
                binding.recyclerTrips.visibility = View.VISIBLE
            }
        }
    }

    private fun openTripDetail(trip: TripEntry) {
        val intent = Intent(this, TripDetailActivity::class.java).apply {
            putExtra(TripDetailActivity.EXTRA_TRIP_ID, trip.id)
        }
        startActivity(intent)
    }

    private fun showDeleteConfirmation(trip: TripEntry) {
        AlertDialog.Builder(this, R.style.NavisunDialogTheme)
            .setTitle(getString(R.string.delete_entry))
            .setMessage(getString(R.string.delete_trip_confirmation))
            .setPositiveButton(getString(R.string.delete)) { _, _ ->
                viewModel.deleteTrip(trip)
            }
            .setNegativeButton(getString(R.string.cancel), null)
            .show()
    }

    private fun checkLocationPermission() {
        if (ContextCompat.checkSelfPermission(
                this, Manifest.permission.ACCESS_FINE_LOCATION
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(
                    Manifest.permission.ACCESS_FINE_LOCATION,
                    Manifest.permission.ACCESS_COARSE_LOCATION
                ),
                LOCATION_PERMISSION_REQUEST
            )
        }
    }
}
