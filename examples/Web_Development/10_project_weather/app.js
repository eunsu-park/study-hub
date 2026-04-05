/*
 * Weather App
 * Weather app using OpenWeatherMap API
 *
 * Note: In production, API keys should be managed via environment variables or server-side.
 * This example uses a free API for learning purposes.
 */

// ============================================
// Configuration
// ============================================
// Free demo API (use your own key for production)
// Get a free key at https://openweathermap.org/api
const API_KEY = 'demo'; // Replace with your actual key
const API_BASE = 'https://api.openweathermap.org/data/2.5/weather';

// Why: Demo mode with sample data allows the example to work out of the box without
// requiring students to sign up for an API key, reducing friction for first-time learners
// Demo mode (uses sample data when no API key is provided)
const DEMO_MODE = API_KEY === 'demo';

// ============================================
// DOM Elements
// ============================================
const cityInput = document.getElementById('cityInput');
const searchBtn = document.getElementById('searchBtn');
const quickCityBtns = document.querySelectorAll('.city-btn');
const loadingEl = document.getElementById('loading');
const errorEl = document.getElementById('error');
const errorMessageEl = document.getElementById('errorMessage');
const weatherDisplay = document.getElementById('weatherDisplay');

// ============================================
// Sample Data (for demo)
// ============================================
const sampleWeatherData = {
    'Seoul': {
        name: 'Seoul',
        sys: { country: 'KR', sunrise: 1706400000, sunset: 1706436000 },
        main: { temp: 3, feels_like: -1, humidity: 45, pressure: 1020 },
        weather: [{ main: 'Clear', description: 'clear sky', icon: '01d' }],
        wind: { speed: 2.5 },
        visibility: 10000,
        clouds: { all: 10 }
    },
    'Tokyo': {
        name: 'Tokyo',
        sys: { country: 'JP', sunrise: 1706396400, sunset: 1706432400 },
        main: { temp: 8, feels_like: 5, humidity: 55, pressure: 1015 },
        weather: [{ main: 'Clouds', description: 'partly cloudy', icon: '02d' }],
        wind: { speed: 3.1 },
        visibility: 8000,
        clouds: { all: 25 }
    },
    'New York': {
        name: 'New York',
        sys: { country: 'US', sunrise: 1706443200, sunset: 1706479200 },
        main: { temp: -2, feels_like: -7, humidity: 60, pressure: 1008 },
        weather: [{ main: 'Snow', description: 'snow', icon: '13d' }],
        wind: { speed: 5.2 },
        visibility: 3000,
        clouds: { all: 90 }
    },
    'London': {
        name: 'London',
        sys: { country: 'GB', sunrise: 1706428800, sunset: 1706461200 },
        main: { temp: 6, feels_like: 3, humidity: 80, pressure: 1012 },
        weather: [{ main: 'Rain', description: 'rain', icon: '10d' }],
        wind: { speed: 4.1 },
        visibility: 6000,
        clouds: { all: 75 }
    },
    'Paris': {
        name: 'Paris',
        sys: { country: 'FR', sunrise: 1706425200, sunset: 1706458800 },
        main: { temp: 5, feels_like: 2, humidity: 70, pressure: 1010 },
        weather: [{ main: 'Clouds', description: 'overcast', icon: '04d' }],
        wind: { speed: 3.5 },
        visibility: 7000,
        clouds: { all: 65 }
    }
};

// ============================================
// Initialize
// ============================================
function init() {
    addEventListeners();

    // Demo mode notification
    if (DEMO_MODE) {
        console.log('Demo mode: Using sample data.');
        console.log('To use the real API, set API_KEY in app.js.');
    }

    // Load initial city
    fetchWeather('Seoul');
}

// ============================================
// Event Listeners
// ============================================
function addEventListeners() {
    // Search button
    searchBtn.addEventListener('click', () => {
        const city = cityInput.value.trim();
        if (city) {
            fetchWeather(city);
        }
    });

    // Enter key
    cityInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            const city = cityInput.value.trim();
            if (city) {
                fetchWeather(city);
            }
        }
    });

    // Quick city buttons
    quickCityBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const city = btn.dataset.city;
            cityInput.value = city;
            fetchWeather(city);
        });
    });
}

// ============================================
// Fetch Weather Data
// ============================================
async function fetchWeather(city) {
    showLoading();
    hideError();
    hideWeather();

    try {
        let data;

        if (DEMO_MODE) {
            // Demo mode: Use sample data
            await simulateDelay(800);
            data = getDemoData(city);
        } else {
            // Real API call
            // Why: encodeURIComponent handles city names with spaces or special characters
            // (e.g., "New York", "Sao Paulo") that would break the URL without encoding
            const url = `${API_BASE}?q=${encodeURIComponent(city)}&appid=${API_KEY}&units=metric&lang=en`;
            const response = await fetch(url);

            if (!response.ok) {
                if (response.status === 404) {
                    throw new Error(`Could not find '${city}'.`);
                } else if (response.status === 401) {
                    throw new Error('API key is not valid.');
                } else {
                    throw new Error('Failed to fetch weather information.');
                }
            }

            data = await response.json();
        }

        displayWeather(data);
    } catch (error) {
        showError(error.message);
    } finally {
        hideLoading();
    }
}

// ============================================
// Demo Data Processing
// ============================================
function getDemoData(city) {
    // Exact name match
    if (sampleWeatherData[city]) {
        return sampleWeatherData[city];
    }

    // Why: Case-insensitive fallback handles user input variations ("seoul", "SEOUL")
    // without requiring exact casing, improving discoverability in the demo
    // Case-insensitive match
    const cityLower = city.toLowerCase();
    for (const [key, value] of Object.entries(sampleWeatherData)) {
        if (key.toLowerCase() === cityLower) {
            return value;
        }
    }

    // Korean city name mapping
    const koreanCities = {
        '\uc11c\uc6b8': 'Seoul',
        '\ub3c4\ucfc4': 'Tokyo',
        '\ub274\uc695': 'New York',
        '\ub7f0\ub358': 'London',
        '\ud30c\ub9ac': 'Paris'
    };

    if (koreanCities[city]) {
        return sampleWeatherData[koreanCities[city]];
    }

    // Not found
    throw new Error(`Could not find '${city}'. In demo mode, only Seoul, Tokyo, New York, London, and Paris are supported.`);
}

function simulateDelay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// ============================================
// Display Weather
// ============================================
function displayWeather(data) {
    // City info
    document.getElementById('cityName').textContent = data.name;
    document.getElementById('country').textContent = getCountryName(data.sys.country);

    // Temperature
    document.getElementById('temp').textContent = Math.round(data.main.temp);

    // Weather icon
    const iconCode = data.weather[0].icon;
    const iconUrl = `https://openweathermap.org/img/wn/${iconCode}@2x.png`;
    document.getElementById('weatherIcon').src = iconUrl;
    document.getElementById('weatherIcon').alt = data.weather[0].description;

    // Description
    document.getElementById('description').textContent = data.weather[0].description;

    // Details
    document.getElementById('feelsLike').textContent = `${Math.round(data.main.feels_like)}\u00b0C`;
    document.getElementById('humidity').textContent = `${data.main.humidity}%`;
    document.getElementById('windSpeed').textContent = `${data.wind.speed} m/s`;
    document.getElementById('pressure').textContent = `${data.main.pressure} hPa`;
    document.getElementById('visibility').textContent = `${(data.visibility / 1000).toFixed(1)} km`;
    document.getElementById('clouds').textContent = `${data.clouds.all}%`;

    // Sunrise/sunset
    document.getElementById('sunrise').textContent = formatTime(data.sys.sunrise);
    document.getElementById('sunset').textContent = formatTime(data.sys.sunset);

    // Update time
    document.getElementById('updateTime').textContent = new Date().toLocaleTimeString('en-US');

    // Change background based on weather
    updateBackground(data.weather[0].main);

    showWeather();
}

// ============================================
// Utility Functions
// ============================================
function formatTime(timestamp) {
    const date = new Date(timestamp * 1000);
    return date.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit'
    });
}

function getCountryName(code) {
    const countries = {
        'KR': 'South Korea',
        'JP': 'Japan',
        'US': 'United States',
        'GB': 'United Kingdom',
        'FR': 'France',
        'CN': 'China',
        'DE': 'Germany',
        'IT': 'Italy',
        'ES': 'Spain',
        'AU': 'Australia'
    };
    return countries[code] || code;
}

function updateBackground(weatherMain) {
    const app = document.querySelector('.app');

    // Why: Removing all weather classes first prevents conflicting styles from stacking
    // when the user searches multiple cities in sequence
    // Remove existing classes
    app.classList.remove('sunny', 'cloudy', 'rainy', 'snowy');

    // Add class based on weather
    switch (weatherMain.toLowerCase()) {
        case 'clear':
            app.classList.add('sunny');
            break;
        case 'clouds':
            app.classList.add('cloudy');
            break;
        case 'rain':
        case 'drizzle':
        case 'thunderstorm':
            app.classList.add('rainy');
            break;
        case 'snow':
            app.classList.add('snowy');
            break;
    }
}

// ============================================
// UI State Management
// ============================================
// Why: Toggling a CSS class instead of style.display preserves animation/transition
// capabilities and keeps presentation logic in CSS where it belongs
function showLoading() {
    loadingEl.classList.remove('hidden');
}

function hideLoading() {
    loadingEl.classList.add('hidden');
}

function showError(message) {
    errorMessageEl.textContent = message;
    errorEl.classList.remove('hidden');
}

function hideError() {
    errorEl.classList.add('hidden');
}

function showWeather() {
    weatherDisplay.classList.remove('hidden');
}

function hideWeather() {
    weatherDisplay.classList.add('hidden');
}

// ============================================
// Start App
// ============================================
init();
