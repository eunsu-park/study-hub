/**
 * Exercise: Component Patterns
 * Practice HOC, render props, compound components, headless components.
 *
 * Setup: npm create vite@latest exercise -- --template react-ts
 */

import React from 'react';

// Exercise 1: Higher-Order Component — withAuth
// Create a HOC that wraps components with auth checking:
// - If user is authenticated, render the wrapped component with user prop
// - If not authenticated, render a login prompt
// - Forward all original props to the wrapped component
// - Preserve the component's displayName for DevTools
// - Add a static property: WrappedComponent.authRequired = true

// TODO: Define WithAuthProps interface
// TODO: Implement withAuth HOC


// Exercise 2: Render Props — DataFetcher
// Create a DataFetcher component using render props pattern:
// - Props: url (string), children (render function)
// - Children receive: { data, loading, error, refetch }
// - Support optional transform prop: (data: unknown) => T
// - Handle AbortController for cleanup
// - Cache responses by URL

// TODO: Implement DataFetcher component
// Usage:
// <DataFetcher url="/api/users">
//   {({ data, loading, error }) => (
//     loading ? <Spinner /> : <UserList users={data} />
//   )}
// </DataFetcher>


// Exercise 3: Compound Component — Accordion
// Build an Accordion using compound component pattern:
// - <Accordion allowMultiple={false}>
//     <Accordion.Item>
//       <Accordion.Header>Section 1</Accordion.Header>
//       <Accordion.Panel>Content 1</Accordion.Panel>
//     </Accordion.Item>
//   </Accordion>
// - Accordion manages which items are open via Context
// - AccordionItem registers itself and provides index
// - AccordionHeader toggles the item
// - AccordionPanel shows/hides content with animation
// - Support allowMultiple and defaultOpen props

// TODO: Implement Accordion compound component


// Exercise 4: Headless Component — useCombobox
// Create a headless combobox (autocomplete) hook:
// - Input: items, onSelect, filterFn, labelExtractor
// - Output: inputProps, listProps, getItemProps, isOpen, highlightedIndex
// - Features: keyboard navigation, filtering, selection, aria attributes
// - The hook provides all behavior; consumers provide all markup
// - Support both controlled and uncontrolled modes

// TODO: Implement useCombobox hook
// TODO: Build a styled example using the hook


// Exercise 5: Slot Pattern — Dashboard Layout
// Create a composable dashboard layout:
// - <DashboardLayout
//     header={<NavBar />}
//     sidebar={<SideMenu items={menuItems} />}
//     toolbar={<ActionBar />}
//     footer={<StatusBar />}
//   >
//     <MainContent />
//   </DashboardLayout>
// - Support collapsible sidebar (toggle button)
// - Responsive: sidebar becomes bottom drawer on mobile
// - Pass layout state (sidebarOpen, isMobile) to children via context
// - Support render function children for access to layout state

// TODO: Implement DashboardLayout component


// --- App to test exercises ---
function App() {
  return (
    <div style={{ maxWidth: 800, margin: '0 auto', padding: 20 }}>
      <h1>Component Patterns Exercises</h1>
      {/* TODO: Render your components here */}
      <p>Implement the exercises above and render them here.</p>
    </div>
  );
}

export default App;
