package com.navisun.fueltracker

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import androidx.core.content.ContextCompat
import com.navisun.fueltracker.service.TripTrackingService

class BootReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action == Intent.ACTION_BOOT_COMPLETED ||
            intent.action == "android.intent.action.QUICKBOOT_POWERON"
        ) {
            // Start tracking service
            val serviceIntent = Intent(context, TripTrackingService::class.java).apply {
                action = TripTrackingService.ACTION_START
            }
            ContextCompat.startForegroundService(context, serviceIntent)

            // Open main activity
            val activityIntent = Intent(context, MainActivity::class.java).apply {
                flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP
            }
            context.startActivity(activityIntent)
        }
    }
}
