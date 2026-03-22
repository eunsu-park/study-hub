/*
 * config.h — Example header demonstrating include guards and macros.
 */

#ifndef CONFIG_H
#define CONFIG_H

#define APP_NAME    "Preprocessor Demo"
#define APP_VERSION "1.0.0"

/* Feature toggles — change to 0 to disable */
#define FEATURE_LOGGING  1
#define FEATURE_DEBUG    1

/* Platform detection */
#if defined(__linux__)
    #define PLATFORM "Linux"
#elif defined(__APPLE__)
    #define PLATFORM "macOS"
#elif defined(_WIN32)
    #define PLATFORM "Windows"
#else
    #define PLATFORM "Unknown"
#endif

#endif /* CONFIG_H */
