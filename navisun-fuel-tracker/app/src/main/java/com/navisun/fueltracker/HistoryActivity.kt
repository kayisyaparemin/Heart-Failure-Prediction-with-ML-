package com.navisun.fueltracker

import android.os.Bundle
import android.view.View
import androidx.activity.viewModels
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
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
                adapter.submitEntriesWithConsumption(entries)
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
}
