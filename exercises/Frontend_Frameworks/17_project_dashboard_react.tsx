/**
 * Exercise: Dashboard Project (React)
 * Build a data dashboard with charts, data fetching, and real-time updates.
 *
 * Setup: npm create vite@latest dashboard -- --template react-ts
 *        npm install recharts zustand
 */

import React from 'react';

// Exercise 1: Dashboard Layout
// Create a responsive dashboard layout:
// - Sidebar: collapsible navigation with icons and labels
// - Header: page title, user avatar, notification bell, search
// - Main area: CSS Grid for widget cards
// - Grid layout: 3 columns desktop, 2 tablet, 1 mobile
// - Sidebar collapse: icon-only on mobile, full on desktop
// - Persist sidebar state in localStorage

// TODO: Implement DashboardLayout component
// TODO: Implement Sidebar component
// TODO: Implement Header component


// Exercise 2: Metric Cards
// Build reusable metric display cards:
// - StatCard: label, value, change (%), trend arrow (up/down)
//   - Green for positive change, red for negative
//   - Animate value counting up on mount
//   - Sparkline mini-chart (last 7 data points)
// - Create 4 cards: Total Users, Revenue, Active Sessions, Conversion Rate
// - Data comes from a useMetrics() hook (simulated API)
// - Show skeleton loader while data loads

// TODO: Implement StatCard component
// TODO: Implement useMetrics hook
// TODO: Implement MetricGrid component


// Exercise 3: Charts
// Implement interactive charts using recharts (or SVG):
// - LineChart: revenue over 30 days, tooltip on hover, zoom
// - BarChart: top 10 pages by visits, horizontal bars
// - PieChart: traffic sources (direct, organic, social, referral)
// - Each chart: loading state, error state, empty state
// - Responsive: charts resize with container
// - Time range selector: 7d, 30d, 90d, 1y

// TODO: Implement RevenueChart component
// TODO: Implement TopPagesChart component
// TODO: Implement TrafficSourcesChart component


// Exercise 4: Data Table
// Build a full-featured data table:
// - Columns: Name, Email, Role, Status, Last Active, Actions
// - Features: sort by column, search filter, pagination (10/25/50)
// - Row actions: view, edit, delete (with confirmation modal)
// - Bulk selection with checkbox column
// - Responsive: horizontal scroll on mobile
// - Loading skeleton rows

// TODO: Implement DataTable component
// TODO: Implement useTableData hook (simulated API with pagination)


// Exercise 5: Real-Time Updates
// Add real-time data simulation:
// - Simulate WebSocket with setInterval (every 5 seconds)
// - Update metric cards with new values (animate transitions)
// - Add new data points to line chart (smooth animation)
// - Show "Last updated: X seconds ago" indicator
// - Activity feed: show recent events in a scrollable list
// - Toast notification on significant changes (> 10% swing)

// TODO: Implement useRealTimeData hook
// TODO: Implement ActivityFeed component
// TODO: Integrate real-time updates into dashboard


function App() {
  return (
    <div>
      <h1>Dashboard Project (React)</h1>
      <p>Build each component from Exercise 1-4, then add real-time updates in Exercise 5.</p>
      {/* TODO: Replace with your composed dashboard */}
    </div>
  );
}

export default App;
