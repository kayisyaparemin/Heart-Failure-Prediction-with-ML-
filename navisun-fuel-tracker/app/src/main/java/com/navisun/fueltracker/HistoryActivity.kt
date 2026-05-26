package com.navisun.fueltracker

import android.os.Bundle
import android.view.View
import androidx.activity.viewModels
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.ItemTouchHelper
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.navisun.fueltracker.adapter.FuelHistoryAdapter
import com.navisun.fueltracker.data.FuelEntry
import com.navisun.fueltracker.databinding.ActivityHistoryBinding
import com.navisun.fueltracker.viewmodel.FuelViewModel
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class HistoryActivity : AppCompatActivity() {

    private lateinit var binding: ActivityHistoryBinding
    private val viewModel: FuelViewModel by viewModels()
    private lateinit var adapter: FuelHistoryAdapter

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityHistoryBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupToolbar()
        setupRecyclerView()
        observeViewModel()
    }

    private fun setupToolbar() {
        setSupportActionBar(binding.toolbar)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        supportActionBar?.title = getString(R.string.history)
        binding.toolbar.setNavigationOnClickListener { finish() }
    }

    private fun setupRecyclerView() {
        adapter = FuelHistoryAdapter { entry ->
            showDeleteConfirmation(entry)
        }

        binding.recyclerView.apply {
            layoutManager = LinearLayoutManager(this@HistoryActivity)
            adapter = this@HistoryActivity.adapter
            setHasFixedSize(false)
        }

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
                val item = adapter.currentList[position]
                // Adapter'ı geri yükle (dialog sonucuna kadar)
                adapter.notifyItemChanged(position)
                showDeleteConfirmation(item.entry)
            }
        })
        itemTouchHelper.attachToRecyclerView(binding.recyclerView)
    }

    private fun observeViewModel() {
        viewModel.allEntries.observe(this) { entries ->
            if (entries.isEmpty()) {
                binding.recyclerView.visibility = View.GONE
                binding.layoutEmpty.visibility = View.VISIBLE
            } else {
                binding.recyclerView.visibility = View.VISIBLE
                binding.layoutEmpty.visibility = View.GONE
                lifecycleScope.launch {
                    val gpsDistances = withContext(Dispatchers.IO) {
                        computeGpsDistances(entries)
                    }
                    adapter.submitEntriesWithConsumption(entries, gpsDistances)
                }
                supportActionBar?.title = getString(R.string.history_with_count, entries.size)
            }
        }
    }

    private fun showDeleteConfirmation(entry: FuelEntry) {
        val dateFormat = SimpleDateFormat("dd.MM.yyyy", Locale("tr", "TR"))
        val dateStr = dateFormat.format(Date(entry.date))

        AlertDialog.Builder(this)
            .setTitle(getString(R.string.delete_entry))
            .setMessage(getString(R.string.delete_confirmation, dateStr))
            .setPositiveButton(getString(R.string.delete)) { _, _ ->
                viewModel.delete(entry)
            }
            .setNegativeButton(getString(R.string.cancel), null)
            .show()
    }

    private suspend fun computeGpsDistances(entries: List<FuelEntry>): Map<Long, Double> {
        val db = com.navisun.fueltracker.data.FuelDatabase.getDatabase(this)
        val result = mutableMapOf<Long, Double>()
        val byType = entries.sortedBy { it.date }.groupBy { it.fuelType }
        for ((_, typeEntries) in byType) {
            val sorted = typeEntries.sortedBy { it.date }
            for (i in 1 until sorted.size) {
                val prev = sorted[i - 1]
                val curr = sorted[i]
                if (curr.fullTank) {
                    val trips = db.tripDao().getTripsBetween(prev.date, curr.date, curr.fuelType)
                    val totalKm = trips.sumOf { it.distanceKm }
                    if (totalKm > 0.5) result[curr.id] = totalKm
                }
            }
        }
        return result
    }
}
