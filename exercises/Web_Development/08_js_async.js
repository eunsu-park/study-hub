/**
 * Exercises for Lesson 08: JavaScript Asynchronous Programming
 * Topic: Web_Development
 * Solutions to practice problems from the lesson.
 * Run: node exercises/Web_Development/08_js_async.js
 *
 * Note: Since we cannot make actual HTTP requests in this standalone file,
 * we simulate API calls with mock functions that return Promises.
 */

// ===== Mock API helpers =====
// Simulates a network delay and returns mock data

function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function mockFetch(endpoint) {
    // Simulate network latency (50-200ms)
    await delay(50 + Math.random() * 150);

    // Route-based mock responses
    const routes = {
        '/api/user/1': { id: 1, name: 'John', email: 'john@example.com' },
        '/api/user/1/posts': [
            { id: 1, title: 'First Post', userId: 1 },
            { id: 2, title: 'Second Post', userId: 1 },
        ],
        '/api/posts/1/comments': [
            { id: 1, text: 'Great post!', postId: 1 },
            { id: 2, text: 'Very helpful.', postId: 1 },
        ],
        '/api/users/1': { id: 1, name: 'Alice', age: 28 },
        '/api/users/2': { id: 2, name: 'Bob', age: 34 },
        '/api/users/3': { id: 3, name: 'Charlie', age: 22 },
        '/api/users/4': { id: 4, name: 'Diana', age: 31 },
        '/api/users/5': { id: 5, name: 'Eve', age: 27 },
    };

    if (routes[endpoint]) {
        return routes[endpoint];
    }
    throw new Error(`404: ${endpoint} not found`);
}


// === Exercise 1: Promise Chaining ===
// Problem: Call 3 APIs sequentially and combine results.
//   /api/user/1         => { name: 'John' }
//   /api/user/1/posts   => [{ id: 1, title: 'First Post' }]
//   /api/posts/1/comments => [{ id: 1, text: 'Great post!' }]

async function getUserDataSequential(userId) {
    // Step 1: Fetch the user
    const user = await mockFetch(`/api/user/${userId}`);

    // Step 2: Fetch the user's posts (depends on user)
    const posts = await mockFetch(`/api/user/${userId}/posts`);

    // Step 3: Fetch comments for the first post (depends on posts)
    const comments = await mockFetch(`/api/posts/${posts[0].id}/comments`);

    // Combine all results
    return {
        user,
        posts,
        firstPostComments: comments,
    };
}

async function exercise1() {
    try {
        const startTime = Date.now();
        const data = await getUserDataSequential(1);
        const elapsed = Date.now() - startTime;

        console.log('  User:', JSON.stringify(data.user));
        console.log('  Posts:', data.posts.length, 'posts found');
        console.log('  Comments on first post:', data.firstPostComments.length, 'comments');
        console.log(`  Total time: ${elapsed}ms (sequential - 3 requests)`);
        console.log('  PASS');
    } catch (error) {
        console.log('  FAIL:', error.message);
    }
}


// === Exercise 2: Parallel Requests ===
// Problem: Fetch information for multiple users simultaneously.
//   const userIds = [1, 2, 3, 4, 5];

async function fetchAllUsers(userIds) {
    // Promise.all: all must succeed
    const promises = userIds.map(id => mockFetch(`/api/users/${id}`));
    return Promise.all(promises);
}

async function fetchAllUsersSafe(userIds) {
    // Promise.allSettled: tolerates individual failures
    const results = await Promise.allSettled(
        userIds.map(id => mockFetch(`/api/users/${id}`))
    );

    return {
        succeeded: results
            .filter(r => r.status === 'fulfilled')
            .map(r => r.value),
        failed: results
            .filter(r => r.status === 'rejected')
            .map(r => r.reason.message),
    };
}

async function exercise2() {
    const userIds = [1, 2, 3, 4, 5];

    // Method 1: Promise.all (all succeed)
    try {
        const startTime = Date.now();
        const users = await fetchAllUsers(userIds);
        const elapsed = Date.now() - startTime;

        console.log('  Method 1 (Promise.all):');
        users.forEach(u => console.log(`    - ${u.name} (age ${u.age})`));
        console.log(`    Time: ${elapsed}ms (parallel - much faster than sequential)`);
    } catch (error) {
        console.log('  Method 1 FAIL:', error.message);
    }

    // Method 2: Promise.allSettled (tolerates failures)
    const userIdsWithBad = [1, 2, 99, 4, 5]; // 99 does not exist
    try {
        const startTime = Date.now();
        const result = await fetchAllUsersSafe(userIdsWithBad);
        const elapsed = Date.now() - startTime;

        console.log('\n  Method 2 (Promise.allSettled) with IDs [1,2,99,4,5]:');
        console.log(`    Succeeded: ${result.succeeded.length} users`);
        result.succeeded.forEach(u => console.log(`      - ${u.name}`));
        console.log(`    Failed: ${result.failed.length} requests`);
        result.failed.forEach(msg => console.log(`      - ${msg}`));
        console.log(`    Time: ${elapsed}ms`);
    } catch (error) {
        console.log('  Method 2 FAIL:', error.message);
    }

    console.log('  PASS');
}


// === Exercise 3: Form Submission (simulated) ===
// Problem: Submit form data to server and handle result.
// Since we have no DOM, we simulate the form submission flow.

async function submitFormData(formData) {
    // Validate
    if (!formData.username || formData.username.length < 3) {
        throw new Error('Username must be at least 3 characters');
    }
    if (!formData.email || !formData.email.includes('@')) {
        throw new Error('Invalid email address');
    }

    // Simulate API call
    await delay(200);

    // Simulate server response
    return {
        success: true,
        message: 'User created successfully',
        user: {
            id: Math.floor(Math.random() * 1000),
            ...formData,
            createdAt: new Date().toISOString(),
        },
    };
}

async function exercise3() {
    // Test 1: Valid submission
    console.log('  Test 1: Valid form data');
    try {
        const result = await submitFormData({
            username: 'johndoe',
            email: 'john@example.com',
            role: 'developer',
        });
        console.log(`    Success: ${result.message}`);
        console.log(`    User ID: ${result.user.id}`);
        console.log(`    Created: ${result.user.createdAt}`);
        console.log('    PASS');
    } catch (error) {
        console.log(`    FAIL: ${error.message}`);
    }

    // Test 2: Invalid username
    console.log('\n  Test 2: Invalid username (too short)');
    try {
        await submitFormData({ username: 'ab', email: 'ab@test.com' });
        console.log('    FAIL: Should have thrown an error');
    } catch (error) {
        console.log(`    Caught expected error: "${error.message}"`);
        console.log('    PASS');
    }

    // Test 3: Invalid email
    console.log('\n  Test 3: Invalid email');
    try {
        await submitFormData({ username: 'alice', email: 'not-an-email' });
        console.log('    FAIL: Should have thrown an error');
    } catch (error) {
        console.log(`    Caught expected error: "${error.message}"`);
        console.log('    PASS');
    }
}


// === Bonus: Retry Logic ===
// Demonstrates the retry pattern from the lesson's practical patterns section.

let attemptCount = 0;

async function unreliableApi() {
    attemptCount++;
    // Fail the first 2 attempts, succeed on the 3rd
    if (attemptCount < 3) {
        throw new Error(`Server error (attempt ${attemptCount})`);
    }
    return { status: 'ok', data: 'Success after retries!' };
}

async function fetchWithRetry(fn, retries = 3, delayMs = 100) {
    for (let i = 0; i < retries; i++) {
        try {
            return await fn();
        } catch (error) {
            if (i === retries - 1) throw error;
            console.log(`    Retry ${i + 1}/${retries}: ${error.message}`);
            await delay(delayMs * (i + 1)); // Exponential backoff
        }
    }
}

async function bonusExercise() {
    attemptCount = 0;
    try {
        const result = await fetchWithRetry(unreliableApi, 3, 50);
        console.log(`    Final result: ${JSON.stringify(result)}`);
        console.log(`    Total attempts: ${attemptCount}`);
        console.log('    PASS');
    } catch (error) {
        console.log(`    FAIL: ${error.message}`);
    }
}


// ===== Run all exercises =====
async function main() {
    console.log('=== Exercise 1: Promise Chaining (Sequential API Calls) ===');
    await exercise1();

    console.log('\n=== Exercise 2: Parallel Requests ===');
    await exercise2();

    console.log('\n=== Exercise 3: Form Submission ===');
    await exercise3();

    console.log('\n=== Bonus: Retry Logic ===');
    await bonusExercise();

    console.log('\nAll exercises completed!');
}

main().catch(console.error);
