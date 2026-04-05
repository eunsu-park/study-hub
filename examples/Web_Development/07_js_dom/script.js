/*
 * DOM Manipulation and Events Examples
 */

// ============================================
// 1. Element Selection
// ============================================
function demoSelection() {
    const output = document.getElementById('selectOutput');
    let result = '';

    // getElementById
    const byId = document.getElementById('uniqueId');
    result += `getElementById: "${byId.textContent}"\n`;

    // getElementsByClassName
    const byClass = document.getElementsByClassName('myClass');
    result += `getElementsByClassName: ${byClass.length} elements\n`;

    // Why: querySelector/All accept any CSS selector, making them far more flexible than
    // getElementById/ClassName; they're the modern go-to for DOM queries
    // querySelector (first match)
    const byQuery = document.querySelector('.myClass');
    result += `querySelector('.myClass'): "${byQuery.textContent}"\n`;

    // querySelectorAll (all matches)
    const byQueryAll = document.querySelectorAll('.myClass');
    result += `querySelectorAll('.myClass'): ${byQueryAll.length} elements\n`;

    // Attribute selector
    const byData = document.querySelector('[data-info="test"]');
    result += `[data-info="test"]: "${byData.textContent}"\n`;

    // Complex selector
    const complex = document.querySelector('#selectDemo .container span');
    result += `Complex selector: "${complex.textContent}"`;

    output.textContent = result;
}

// ============================================
// 2. Creating and Adding Elements
// ============================================
function addItem() {
    const input = document.getElementById('newItemInput');
    const list = document.getElementById('itemList');
    const text = input.value.trim();

    if (!text) return;

    // Create element
    const li = document.createElement('li');
    li.textContent = text;

    // Add delete button
    const deleteBtn = document.createElement('button');
    deleteBtn.textContent = '\u00d7';
    deleteBtn.style.cssText = 'margin-left: 10px; padding: 2px 8px; background: #e74c3c;';
    deleteBtn.onclick = () => li.remove();

    li.appendChild(deleteBtn);
    list.appendChild(li);  // Add to end

    input.value = '';
    input.focus();
}

function addItemBefore() {
    const input = document.getElementById('newItemInput');
    const list = document.getElementById('itemList');
    const text = input.value.trim();

    if (!text) return;

    const li = document.createElement('li');
    li.textContent = text;

    const deleteBtn = document.createElement('button');
    deleteBtn.textContent = '\u00d7';
    deleteBtn.style.cssText = 'margin-left: 10px; padding: 2px 8px; background: #e74c3c;';
    deleteBtn.onclick = () => li.remove();

    li.appendChild(deleteBtn);

    // Add to beginning
    list.insertBefore(li, list.firstChild);

    input.value = '';
}

function clearItems() {
    const list = document.getElementById('itemList');
    // Remove all children
    list.innerHTML = '';
    // Or: while (list.firstChild) list.removeChild(list.firstChild);
}

// Add with Enter key
document.getElementById('newItemInput').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') addItem();
});

// ============================================
// 3. Style and Class Manipulation
// ============================================
function changeColor() {
    const box = document.getElementById('styleBox');
    const colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12'];
    const randomColor = colors[Math.floor(Math.random() * colors.length)];
    box.style.backgroundColor = randomColor;
    box.style.color = 'white';
}

// Why: classList.toggle is safer than manually checking/adding/removing classes,
// avoiding duplicate class entries and reducing code from 4 lines to 1
function toggleHighlight() {
    const box = document.getElementById('styleBox');
    box.classList.toggle('highlight');
}

function addBorder() {
    const box = document.getElementById('styleBox');
    box.style.border = '3px solid #2c3e50';
    box.style.borderRadius = '15px';
}

function resetStyle() {
    const box = document.getElementById('styleBox');
    box.style.cssText = 'transition: all 0.3s;';
    box.classList.remove('highlight');
}

// ============================================
// 4. Attribute Manipulation
// ============================================
function changeImage(num) {
    const img = document.getElementById('demoImage');
    img.src = `https://via.placeholder.com/200x100?text=Image+${num}`;
    img.alt = `Demo image ${num}`;
    updateAltDisplay();
}

function changeAlt() {
    const img = document.getElementById('demoImage');
    const altInput = document.getElementById('altInput');
    img.setAttribute('alt', altInput.value);
    updateAltDisplay();
    altInput.value = '';
}

function updateAltDisplay() {
    const img = document.getElementById('demoImage');
    document.getElementById('currentAlt').textContent = img.getAttribute('alt');
}

// Initial alt display
updateAltDisplay();

// ============================================
// 5. Event Listeners
// ============================================
const eventBox = document.getElementById('eventBox');
const eventLog = document.getElementById('eventLog');

function logEvent(message) {
    const time = new Date().toLocaleTimeString();
    eventLog.innerHTML = `[${time}] ${message}<br>` + eventLog.innerHTML;
}

function clearEventLog() {
    eventLog.innerHTML = '';
}

// Click event
eventBox.addEventListener('click', (e) => {
    logEvent(`Click! Coordinates: (${e.offsetX}, ${e.offsetY})`);
});

// Double click
eventBox.addEventListener('dblclick', () => {
    logEvent('Double click!');
});

// Mouse enter/leave
eventBox.addEventListener('mouseenter', () => {
    logEvent('Mouse enter');
    eventBox.style.backgroundColor = '#ecf0f1';
});

eventBox.addEventListener('mouseleave', () => {
    logEvent('Mouse leave');
    eventBox.style.backgroundColor = '';
});

// Why: Manual throttle prevents mousemove from flooding the log (fires 60+ times/sec),
// keeping the demo readable while still demonstrating the event
// Mouse move (throttled)
let lastMoveLog = 0;
eventBox.addEventListener('mousemove', (e) => {
    const now = Date.now();
    if (now - lastMoveLog > 500) {  // Log every 500ms
        logEvent(`Mouse move: (${e.offsetX}, ${e.offsetY})`);
        lastMoveLog = now;
    }
});

// Context menu (right click)
eventBox.addEventListener('contextmenu', (e) => {
    e.preventDefault();
    logEvent('Right click (default action prevented)');
});

// Why: Event delegation attaches a single listener to the parent instead of one per card,
// reducing memory usage and automatically handling dynamically added cards
// ============================================
// 6. Event Delegation
// ============================================
const cardContainer = document.getElementById('cardContainer');
let cardCount = 5;

cardContainer.addEventListener('click', (e) => {
    const card = e.target.closest('.card');
    if (!card) return;

    // Remove selected from all cards
    cardContainer.querySelectorAll('.card').forEach(c => c.classList.remove('selected'));

    // Add selected to clicked card
    card.classList.add('selected');

    document.getElementById('selectedCard').textContent = `Card ${card.dataset.id}`;
});

function addCard() {
    cardCount++;
    const card = document.createElement('div');
    card.className = 'card';
    card.dataset.id = cardCount;
    card.textContent = cardCount;
    cardContainer.appendChild(card);
}

// ============================================
// 7. Form Events
// ============================================
const demoForm = document.getElementById('demoForm');
const formOutput = document.getElementById('formOutput');

// input event (real-time)
document.getElementById('nameInput').addEventListener('input', (e) => {
    console.log('Typing:', e.target.value);
});

// change event (on blur)
document.getElementById('countrySelect').addEventListener('change', (e) => {
    formOutput.textContent = `Country changed: ${e.target.value || 'Not selected'}`;
});

// focus/blur events
document.getElementById('emailInput').addEventListener('focus', (e) => {
    e.target.style.borderColor = '#3498db';
});

document.getElementById('emailInput').addEventListener('blur', (e) => {
    e.target.style.borderColor = '';
});

// Why: preventDefault on submit stops the browser from reloading the page, letting JS
// handle validation and async submission for a smoother user experience
// submit event
demoForm.addEventListener('submit', (e) => {
    e.preventDefault();  // Prevent default action

    const formData = new FormData(demoForm);
    const name = document.getElementById('nameInput').value;
    const email = document.getElementById('emailInput').value;
    const country = document.getElementById('countrySelect').value;

    formOutput.innerHTML = `
        <strong>Submitted data:</strong><br>
        Name: ${name}<br>
        Email: ${email}<br>
        Country: ${country || 'Not selected'}
    `;
});

// ============================================
// 8. Keyboard Events
// ============================================
const keyInput = document.getElementById('keyInput');
const keyOutput = document.getElementById('keyOutput');

keyInput.addEventListener('keydown', (e) => {
    keyOutput.innerHTML = `
        <strong>keydown</strong><br>
        key: "${e.key}"<br>
        code: "${e.code}"<br>
        keyCode: ${e.keyCode} (deprecated)<br>
        ctrlKey: ${e.ctrlKey}, shiftKey: ${e.shiftKey}, altKey: ${e.altKey}
    `;

    // Special key handling
    if (e.key === 'Escape') {
        keyInput.value = '';
        keyOutput.innerHTML += '<br><em>Input cleared with Escape</em>';
    }

    if (e.ctrlKey && e.key === 'Enter') {
        keyOutput.innerHTML += '<br><em>Ctrl+Enter detected!</em>';
    }
});

// ============================================
// 9. Drag and Drop
// ============================================
let draggedElement = null;

// Draggable elements
document.querySelectorAll('.draggable').forEach(elem => {
    elem.addEventListener('dragstart', (e) => {
        draggedElement = e.target;
        e.target.style.opacity = '0.5';
    });

    elem.addEventListener('dragend', (e) => {
        e.target.style.opacity = '';
        draggedElement = null;
    });
});

// Why: The browser's default behavior for dragover is to reject drops, so
// preventDefault() is required to make an element a valid drop target
// Drop zones
document.querySelectorAll('.drop-zone').forEach(zone => {
    zone.addEventListener('dragover', (e) => {
        e.preventDefault();  // Allow drop
        zone.classList.add('drag-over');
    });

    zone.addEventListener('dragleave', () => {
        zone.classList.remove('drag-over');
    });

    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('drag-over');

        if (draggedElement) {
            zone.appendChild(draggedElement);
        }
    });
});

// ============================================
// 10. Scroll Events
// ============================================
const scrollBox = document.getElementById('scrollBox');
const scrollPosition = document.getElementById('scrollPosition');

scrollBox.addEventListener('scroll', () => {
    scrollPosition.textContent = Math.round(scrollBox.scrollTop);
});

// Scroll direction detection (bonus)
let lastScrollTop = 0;
scrollBox.addEventListener('scroll', () => {
    const st = scrollBox.scrollTop;
    const direction = st > lastScrollTop ? 'down' : 'up';
    // console.log(`Scroll direction: ${direction}`);
    lastScrollTop = st;
});

// ============================================
// Page Load Complete
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded!');
});

window.addEventListener('load', () => {
    console.log('All resources loaded!');
});
