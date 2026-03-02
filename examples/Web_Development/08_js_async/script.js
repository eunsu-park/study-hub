/*
 * Asynchronous JavaScript Examples
 * - Callbacks, Promises, async/await
 * - Fetch API
 * - Timer functions
 */

// ============================================
// Utility Functions
// ============================================
function log(outputId, message, clear = false) {
    const output = document.getElementById(outputId);
    if (clear) {
        output.textContent = '';
    }
    output.textContent += message + '\n';
    output.scrollTop = output.scrollHeight;
}

function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// ============================================
// 1. Callback Functions
// ============================================
function runCallback() {
    const output = 'callbackOutput';
    log(output, '=== Running Callback Function ===', true);
    log(output, '[Start] Requesting data...');

    function fetchData(callback) {
        setTimeout(() => {
            const data = { id: 1, name: 'John Doe', email: 'john@example.com' };
            callback(null, data);
        }, 1000);
    }

    fetchData((error, data) => {
        if (error) {
            log(output, `[Error] ${error.message}`);
        } else {
            log(output, `[Done] Data received: ${JSON.stringify(data, null, 2)}`);
        }
    });

    log(output, '[Async] Performing other tasks...');
}

function runCallbackHell() {
    const output = 'callbackOutput';
    log(output, '=== Callback Hell Example ===', true);
    log(output, 'Starting user info lookup...');

    // Why: Nested callbacks create rightward drift that is hard to read, test, and debug -
    // this "pyramid of doom" motivates the switch to Promises and async/await
    // Callback hell (pyramid shape)
    setTimeout(() => {
        log(output, '1. User ID lookup complete: user_123');
        setTimeout(() => {
            log(output, '2. User profile lookup complete: { name: "John Doe" }');
            setTimeout(() => {
                log(output, '3. User posts lookup complete: 15 posts');
                setTimeout(() => {
                    log(output, '4. Post comments lookup complete: 42 comments');
                    log(output, '--- All tasks complete ---');
                    log(output, '\nThis nested structure hurts readability.');
                    log(output, 'Use Promises or async/await to improve this!');
                }, 500);
            }, 500);
        }, 500);
    }, 500);
}

// ============================================
// 2. Promise
// ============================================
function runPromise() {
    const output = 'promiseOutput';
    log(output, '=== Promise Basics ===', true);

    const promise = new Promise((resolve, reject) => {
        log(output, 'Starting task...');
        setTimeout(() => {
            const success = Math.random() > 0.3;
            if (success) {
                resolve({ status: 'success', data: 'Processing complete!' });
            } else {
                reject(new Error('Processing failed'));
            }
        }, 1000);
    });

    promise
        .then(result => {
            log(output, `Success: ${JSON.stringify(result)}`);
        })
        .catch(error => {
            log(output, `Failed: ${error.message}`);
        })
        .finally(() => {
            log(output, 'Promise finished (regardless of success/failure)');
        });
}

function runPromiseChain() {
    const output = 'promiseOutput';
    log(output, '=== Promise Chaining ===', true);

    function step1() {
        return new Promise(resolve => {
            setTimeout(() => resolve(1), 300);
        });
    }

    function step2(prev) {
        return new Promise(resolve => {
            setTimeout(() => resolve(prev + 10), 300);
        });
    }

    function step3(prev) {
        return new Promise(resolve => {
            setTimeout(() => resolve(prev * 2), 300);
        });
    }

    log(output, 'Chaining started...');

    step1()
        .then(result => {
            log(output, `Step 1: ${result}`);
            return step2(result);
        })
        .then(result => {
            log(output, `Step 2: ${result}`);
            return step3(result);
        })
        .then(result => {
            log(output, `Step 3: ${result}`);
            log(output, `Final result: ${result}`);
        });
}

function runPromiseError() {
    const output = 'promiseOutput';
    log(output, '=== Promise Error Handling ===', true);

    function riskyOperation(shouldFail) {
        return new Promise((resolve, reject) => {
            setTimeout(() => {
                if (shouldFail) {
                    reject(new Error('Intentional error!'));
                } else {
                    resolve('Successfully processed');
                }
            }, 500);
        });
    }

    log(output, 'Running error scenario...');

    riskyOperation(true)
        .then(result => {
            log(output, `Result: ${result}`);
        })
        .catch(error => {
            log(output, `Error caught: ${error.message}`);
            log(output, 'Error was handled in the catch block');
            return 'Recovered from error';
        })
        .then(result => {
            log(output, `After recovery: ${result}`);
        });
}

// ============================================
// 3. async/await
// ============================================
async function runAsyncAwait() {
    const output = 'asyncOutput';
    log(output, '=== async/await Basics ===', true);

    async function fetchUserData() {
        log(output, 'Requesting user data...');
        await delay(500);
        return { id: 1, name: 'Jane Doe' };
    }

    async function fetchUserPosts(userId) {
        log(output, `Requesting posts for user ${userId}...`);
        await delay(500);
        return ['Post 1', 'Post 2', 'Post 3'];
    }

    try {
        const user = await fetchUserData();
        log(output, `User: ${JSON.stringify(user)}`);

        const posts = await fetchUserPosts(user.id);
        log(output, `Posts: ${posts.join(', ')}`);

        log(output, 'All tasks complete!');
    } catch (error) {
        log(output, `Error: ${error.message}`);
    }
}

async function runAsyncError() {
    const output = 'asyncOutput';
    log(output, '=== async/await Error Handling ===', true);

    async function unstableOperation() {
        await delay(500);
        throw new Error('Network connection failed');
    }

    log(output, 'Attempting unstable operation...');

    try {
        await unstableOperation();
        log(output, 'Operation succeeded');
    } catch (error) {
        log(output, `Error caught with try-catch: ${error.message}`);
        log(output, 'Use try-catch with async/await');
    } finally {
        log(output, 'finally block executed');
    }
}

// ============================================
// 4. Promise.all, Promise.race, Promise.allSettled
// ============================================
async function runPromiseAll() {
    const output = 'promiseAllOutput';
    log(output, '=== Promise.all ===', true);
    log(output, 'Running 3 tasks in parallel...');

    const startTime = Date.now();

    // Why: Creating promises first, then awaiting Promise.all runs tasks concurrently -
    // if we awaited each sequentially, total time would be 3.3s instead of ~1.5s
    const task1 = delay(1000).then(() => 'Task 1 complete');
    const task2 = delay(1500).then(() => 'Task 2 complete');
    const task3 = delay(800).then(() => 'Task 3 complete');

    try {
        const results = await Promise.all([task1, task2, task3]);
        const elapsed = Date.now() - startTime;

        log(output, `\nResults: ${results.join(', ')}`);
        log(output, `Total elapsed time: ${elapsed}ms`);
        log(output, '\nPromise.all waits until all Promises are fulfilled');
        log(output, 'Completes based on the longest task (1500ms)');
    } catch (error) {
        log(output, `Error: ${error.message}`);
    }
}

async function runPromiseRace() {
    const output = 'promiseAllOutput';
    log(output, '=== Promise.race ===', true);
    log(output, 'Returns the result of the fastest task...');

    const startTime = Date.now();

    const slow = delay(2000).then(() => 'Slow task');
    const medium = delay(1000).then(() => 'Medium task');
    const fast = delay(500).then(() => 'Fast task');

    const winner = await Promise.race([slow, medium, fast]);
    const elapsed = Date.now() - startTime;

    log(output, `\nWinner: ${winner}`);
    log(output, `Elapsed time: ${elapsed}ms`);
    log(output, '\nPromise.race returns the first Promise to settle');
}

async function runPromiseAllSettled() {
    const output = 'promiseAllOutput';
    log(output, '=== Promise.allSettled ===', true);
    log(output, 'Collects all results regardless of success/failure...');

    const tasks = [
        delay(500).then(() => 'Success 1'),
        delay(300).then(() => { throw new Error('Failure 1'); }),
        delay(400).then(() => 'Success 2'),
        delay(600).then(() => { throw new Error('Failure 2'); }),
    ];

    const results = await Promise.allSettled(tasks);

    log(output, '\nResults:');
    results.forEach((result, index) => {
        if (result.status === 'fulfilled') {
            log(output, `  ${index + 1}. ${result.value}`);
        } else {
            log(output, `  ${index + 1}. ${result.reason.message}`);
        }
    });

    log(output, '\nPromise.allSettled collects results from all Promises');
    log(output, 'You can check remaining results even if some fail');
}

// ============================================
// 5. Fetch API
// ============================================
async function fetchUsers() {
    const output = 'fetchOutput';
    const btn = document.getElementById('fetchUsersBtn');

    log(output, '=== Fetch User List ===', true);
    btn.disabled = true;
    btn.innerHTML = 'Loading... <span class="loading"></span>';

    try {
        const response = await fetch('https://jsonplaceholder.typicode.com/users?_limit=5');

        // Why: fetch only rejects on network failure, not HTTP errors; checking response.ok
        // catches 4xx/5xx status codes that would otherwise silently return invalid data
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const users = await response.json();

        log(output, `Found ${users.length} users:\n`);

        users.forEach(user => {
            log(output, `${user.name}`);
            log(output, `   Email: ${user.email}`);
            log(output, `   Company: ${user.company.name}`);
            log(output, '');
        });
    } catch (error) {
        log(output, `Error: ${error.message}`);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Fetch User List';
    }
}

async function fetchPost() {
    const output = 'fetchOutput';
    log(output, '=== Fetch Post ===', true);

    const postId = Math.floor(Math.random() * 100) + 1;
    log(output, `Requesting post #${postId}...`);

    try {
        const response = await fetch(`https://jsonplaceholder.typicode.com/posts/${postId}`);
        const post = await response.json();

        log(output, `\nTitle: ${post.title}`);
        log(output, `\nBody:\n${post.body}`);
    } catch (error) {
        log(output, `Error: ${error.message}`);
    }
}

// ============================================
// 6. POST Request
// ============================================
async function createPost() {
    const output = 'postOutput';
    const title = document.getElementById('postTitle').value || 'No title';
    const body = document.getElementById('postBody').value || 'No content';

    log(output, '=== POST Request ===', true);
    log(output, 'Creating post...');

    try {
        const response = await fetch('https://jsonplaceholder.typicode.com/posts', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                title: title,
                body: body,
                userId: 1,
            }),
        });

        const data = await response.json();

        log(output, '\nPost created successfully!');
        log(output, `Response data:`);
        log(output, JSON.stringify(data, null, 2));
    } catch (error) {
        log(output, `Error: ${error.message}`);
    }
}

// ============================================
// 7. Error Handling Patterns
// ============================================
async function testErrorHandling() {
    const output = 'errorOutput';
    log(output, '=== Error Handling Patterns ===', true);

    // Pattern 1: try-catch
    log(output, '1. try-catch pattern:');
    try {
        const result = await fetch('https://jsonplaceholder.typicode.com/posts/1');
        const data = await result.json();
        log(output, `   Success: ${data.title.substring(0, 20)}...`);
    } catch (error) {
        log(output, `   Failed: ${error.message}`);
    }

    // Pattern 2: Error wrapping
    log(output, '\n2. Error wrapping pattern:');

    // Why: Returning {data, error} tuples instead of throwing makes error handling explicit
    // at call sites and avoids deeply nested try-catch blocks in calling code
    async function safeFetch(url) {
        try {
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            return { data: await response.json(), error: null };
        } catch (error) {
            return { data: null, error: error.message };
        }
    }

    const { data, error } = await safeFetch('https://jsonplaceholder.typicode.com/posts/1');
    if (error) {
        log(output, `   Error: ${error}`);
    } else {
        log(output, `   Data: ${data.title.substring(0, 20)}...`);
    }

    log(output, '\nThe error wrapping pattern is Go-style');
    log(output, 'It enables explicit error handling');
}

async function testNetworkError() {
    const output = 'errorOutput';
    log(output, '=== Network Error Test ===', true);
    log(output, 'Requesting non-existent URL...\n');

    try {
        // Why: AbortController provides a standard way to cancel fetch requests,
        // preventing memory leaks and stale responses when the user navigates away or a timeout fires
        // Timeout implementation
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 3000);

        const response = await fetch('https://invalid-url-that-does-not-exist.com/api', {
            signal: controller.signal,
        });

        clearTimeout(timeoutId);
        const data = await response.json();
        log(output, `Success: ${data}`);
    } catch (error) {
        if (error.name === 'AbortError') {
            log(output, 'Timeout: No response within 3 seconds');
        } else {
            log(output, `Network error: ${error.message}`);
        }
        log(output, '\nIn real apps, adding retry logic is recommended');
    }
}

// ============================================
// 8. Load User Cards
// ============================================
async function loadUserCards() {
    const container = document.getElementById('userCards');
    const btn = document.getElementById('loadCardsBtn');

    container.innerHTML = '<p>Loading...</p>';
    btn.disabled = true;

    try {
        const response = await fetch('https://jsonplaceholder.typicode.com/users?_limit=6');
        const users = await response.json();

        container.innerHTML = users.map(user => `
            <div class="user-card">
                <img src="https://i.pravatar.cc/80?u=${user.id}" alt="${user.name}">
                <h4>${user.name}</h4>
                <p>@${user.username}</p>
                <p>${user.email}</p>
            </div>
        `).join('');
    } catch (error) {
        container.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
    } finally {
        btn.disabled = false;
    }
}

// ============================================
// 9. setTimeout, setInterval
// ============================================
let timerCount = 0;
let timerInterval = null;

function startTimer() {
    const output = document.getElementById('timerOutput');

    if (timerInterval) {
        output.textContent = 'Timer is already running.';
        return;
    }

    timerCount = 0;
    output.textContent = `Timer: ${timerCount}`;

    timerInterval = setInterval(() => {
        timerCount++;
        output.textContent = `Timer: ${timerCount}s`;

        if (timerCount >= 60) {
            stopTimer();
            output.textContent += ' (max time reached)';
        }
    }, 1000);
}

function stopTimer() {
    const output = document.getElementById('timerOutput');

    if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
        output.textContent = `Timer stopped at ${timerCount}s`;
    } else {
        output.textContent = 'No timer is running.';
    }
}

// Why: Debounce delays execution until the user stops triggering events, preventing
// expensive operations (API calls, DOM updates) from firing on every keystroke or scroll
// Debounce implementation
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function runDebounceDemo() {
    const output = document.getElementById('timerOutput');
    let callCount = 0;

    const debouncedLog = debounce(() => {
        output.textContent = `Debounce complete! Actual executions: 1 (attempts: ${callCount})`;
    }, 500);

    output.textContent = 'Simulating rapid calls...';

    // Attempt 10 calls at 100ms intervals
    for (let i = 0; i < 10; i++) {
        setTimeout(() => {
            callCount++;
            debouncedLog();
        }, i * 100);
    }

    setTimeout(() => {
        output.textContent += '\nDebounce: Executes after 500ms wait since last call';
    }, 2000);
}
