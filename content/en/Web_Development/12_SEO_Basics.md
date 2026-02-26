# 12. SEO Basics (Search Engine Optimization)

**Previous**: [Web Accessibility](./11_Web_Accessibility.md) | **Next**: [Build Tools & Development Environment](./13_Build_Tools_Environment.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Describe how search engines crawl, index, and rank web pages
2. Write effective title tags, meta descriptions, and canonical URLs for on-page SEO
3. Implement Open Graph and Twitter Card meta tags for social media sharing
4. Add JSON-LD structured data for articles, products, FAQs, and breadcrumbs
5. Configure `robots.txt` and `sitemap.xml` to control crawler behavior
6. Optimize page speed by applying Core Web Vitals best practices
7. Evaluate a site's SEO health using Google Search Console and Lighthouse

---

Building a great website means nothing if no one can find it. Search engines are the primary way users discover new content, and SEO is the practice of making your pages visible to those search engines. Understanding SEO is not just a marketing skill -- it directly influences how you structure HTML, write content, and architect your site's technical foundation.

## Table of Contents
1. [SEO Overview](#1-seo-overview)
2. [Meta Tags](#2-meta-tags)
3. [Structured Data](#3-structured-data)
4. [Technical SEO](#4-technical-seo)
5. [Content SEO](#5-content-seo)
6. [Measurement and Tools](#6-measurement-and-tools)
7. [Practice Problems](#7-practice-problems)

---

## 1. SEO Overview

### 1.1 What is SEO?

```
┌─────────────────────────────────────────────────────────────────┐
│                    SEO Definition and Purpose                    │
│                                                                 │
│   SEO (Search Engine Optimization):                            │
│   The process of improving website visibility on search        │
│   engine results pages (SERP)                                  │
│                                                                 │
│   Goals:                                                        │
│   - Increase organic traffic                                   │
│   - Enhance brand awareness                                    │
│   - Build trust                                                │
│   - Improve conversion rate                                    │
│                                                                 │
│   How Search Engines Work:                                     │
│   1. Crawling: Discover web pages                             │
│   2. Indexing: Analyze and store content                      │
│   3. Ranking: Determine order for search queries              │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Types of SEO

```
┌─────────────────────────────────────────────────────────────────┐
│                    SEO Type Classification                       │
│                                                                 │
│   On-page SEO:                                                  │
│   - Meta tags (title, description)                             │
│   - Content quality                                            │
│   - Keyword optimization                                       │
│   - Image optimization                                         │
│   - Internal links                                             │
│                                                                 │
│   Off-page SEO:                                                 │
│   - Backlinks (links from other sites)                         │
│   - Social signals                                             │
│   - Brand mentions                                             │
│   - Guest posting                                              │
│                                                                 │
│   Technical SEO:                                                │
│   - Site speed                                                 │
│   - Mobile-friendliness                                        │
│   - HTTPS                                                      │
│   - Structured data                                            │
│   - Crawling/indexing control                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Meta Tags

### 2.1 Essential Meta Tags

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <!-- Character encoding -->
  <meta charset="UTF-8">

  <!-- Responsive viewport -->
  <meta name="viewport" content="width=device-width, initial-scale=1.0">

  <!-- Page title (50-60 characters recommended) -->
  <title>Web Development Basics Guide - Complete Tutorial for Beginners | SiteName</title>

  <!-- Meta description (150-160 characters recommended) -->
  <meta name="description" content="From HTML, CSS, JavaScript basics to practical applications.
    Step-by-step guide for web development beginners. Learn with free example code.">

  <!-- Search engine instructions -->
  <meta name="robots" content="index, follow">

  <!-- Canonical URL (prevent duplicate content) -->
  <link rel="canonical" href="https://example.com/web-development-guide">
</head>
</html>
```

### 2.2 Open Graph (Social Media)

```html
<!-- Facebook / LinkedIn -->
<meta property="og:type" content="article">
<meta property="og:title" content="Web Development Basics Guide">
<meta property="og:description" content="Complete web development tutorial for beginners">
<meta property="og:image" content="https://example.com/images/og-image.jpg">
<meta property="og:url" content="https://example.com/web-development-guide">
<meta property="og:site_name" content="SiteName">
<meta property="og:locale" content="en_US">

<!-- Recommended image size: 1200x630px -->
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:image:alt" content="Web development guide thumbnail">
```

### 2.3 Twitter Card

```html
<!-- Twitter -->
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:site" content="@username">
<meta name="twitter:creator" content="@author">
<meta name="twitter:title" content="Web Development Basics Guide">
<meta name="twitter:description" content="Complete web development tutorial for beginners">
<meta name="twitter:image" content="https://example.com/images/twitter-image.jpg">

<!-- Card types: summary, summary_large_image, app, player -->
```

### 2.4 Other Important Meta Tags

```html
<!-- Language/region settings -->
<link rel="alternate" hreflang="en" href="https://example.com/en/">
<link rel="alternate" hreflang="ko" href="https://example.com/ko/">
<link rel="alternate" hreflang="x-default" href="https://example.com/">

<!-- Favicon -->
<link rel="icon" type="image/x-icon" href="/favicon.ico">
<link rel="apple-touch-icon" href="/apple-touch-icon.png">

<!-- Theme color (mobile browser) -->
<meta name="theme-color" content="#4285f4">

<!-- Author information -->
<meta name="author" content="Author Name">

<!-- Publication/modification date -->
<meta property="article:published_time" content="2024-01-15T09:00:00+00:00">
<meta property="article:modified_time" content="2024-03-20T14:30:00+00:00">
```

---

## 3. Structured Data

### 3.1 JSON-LD Basics

```html
<!-- JSON-LD structured data (recommended format) -->
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "WebPage",
  "name": "Web Development Basics Guide",
  "description": "Complete web development tutorial for beginners",
  "url": "https://example.com/web-development-guide"
}
</script>
```

### 3.2 Organization Information

```html
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "Organization",
  "name": "Company Name",
  "url": "https://example.com",
  "logo": "https://example.com/logo.png",
  "sameAs": [
    "https://www.facebook.com/company",
    "https://www.twitter.com/company",
    "https://www.linkedin.com/company/company"
  ],
  "contactPoint": {
    "@type": "ContactPoint",
    "telephone": "+1-555-1234",
    "contactType": "customer service",
    "availableLanguage": ["English", "Korean"]
  }
}
</script>
```

### 3.3 Article/Blog Post

```html
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "Article",
  "headline": "Web Development Basics Guide",
  "description": "Complete web development tutorial for beginners",
  "image": [
    "https://example.com/images/1x1.jpg",
    "https://example.com/images/4x3.jpg",
    "https://example.com/images/16x9.jpg"
  ],
  "author": {
    "@type": "Person",
    "name": "Author Name",
    "url": "https://example.com/author"
  },
  "publisher": {
    "@type": "Organization",
    "name": "SiteName",
    "logo": {
      "@type": "ImageObject",
      "url": "https://example.com/logo.png"
    }
  },
  "datePublished": "2024-01-15",
  "dateModified": "2024-03-20",
  "mainEntityOfPage": {
    "@type": "WebPage",
    "@id": "https://example.com/web-development-guide"
  }
}
</script>
```

### 3.4 FAQ Page

```html
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "FAQPage",
  "mainEntity": [
    {
      "@type": "Question",
      "name": "What is HTML?",
      "acceptedAnswer": {
        "@type": "Answer",
        "text": "HTML (HyperText Markup Language) is a markup language that defines the structure of web pages."
      }
    },
    {
      "@type": "Question",
      "name": "What is CSS used for?",
      "acceptedAnswer": {
        "@type": "Answer",
        "text": "CSS (Cascading Style Sheets) is used to define the visual styling of web pages."
      }
    }
  ]
}
</script>
```

### 3.5 Product Information

```html
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "Product",
  "name": "Product Name",
  "image": "https://example.com/product.jpg",
  "description": "Product description",
  "sku": "SKU12345",
  "brand": {
    "@type": "Brand",
    "name": "Brand Name"
  },
  "offers": {
    "@type": "Offer",
    "url": "https://example.com/product",
    "priceCurrency": "USD",
    "price": "99.00",
    "availability": "https://schema.org/InStock",
    "seller": {
      "@type": "Organization",
      "name": "Seller Name"
    }
  },
  "aggregateRating": {
    "@type": "AggregateRating",
    "ratingValue": "4.5",
    "reviewCount": "128"
  }
}
</script>
```

### 3.6 Breadcrumbs

```html
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "BreadcrumbList",
  "itemListElement": [
    {
      "@type": "ListItem",
      "position": 1,
      "name": "Home",
      "item": "https://example.com"
    },
    {
      "@type": "ListItem",
      "position": 2,
      "name": "Tutorials",
      "item": "https://example.com/tutorials"
    },
    {
      "@type": "ListItem",
      "position": 3,
      "name": "Web Development",
      "item": "https://example.com/tutorials/web-development"
    }
  ]
}
</script>
```

---

## 4. Technical SEO

### 4.1 Sitemap (sitemap.xml)

```xml
<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://example.com/</loc>
    <lastmod>2024-03-20</lastmod>
    <changefreq>daily</changefreq>
    <priority>1.0</priority>
  </url>
  <url>
    <loc>https://example.com/about</loc>
    <lastmod>2024-02-15</lastmod>
    <changefreq>monthly</changefreq>
    <priority>0.8</priority>
  </url>
  <url>
    <loc>https://example.com/blog/post-1</loc>
    <lastmod>2024-03-18</lastmod>
    <changefreq>weekly</changefreq>
    <priority>0.6</priority>
  </url>
</urlset>
```

### 4.2 robots.txt

```
# robots.txt
User-agent: *
Allow: /
Disallow: /admin/
Disallow: /private/
Disallow: /api/
Disallow: /*?*  # Exclude URL parameters

# Sitemap location
Sitemap: https://example.com/sitemap.xml

# Specific bot settings
User-agent: Googlebot
Allow: /
Crawl-delay: 1

User-agent: Bingbot
Crawl-delay: 5
```

### 4.3 Page Speed Optimization

```html
<!-- Resource hints -->
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="dns-prefetch" href="https://cdn.example.com">
<link rel="preload" href="/fonts/main.woff2" as="font" type="font/woff2" crossorigin>

<!-- Image optimization -->
<img
  src="image.webp"
  alt="Description"
  width="800"
  height="600"
  loading="lazy"
  decoding="async"
>

<!-- Responsive images -->
<picture>
  <source media="(min-width: 800px)" srcset="large.webp">
  <source media="(min-width: 400px)" srcset="medium.webp">
  <img src="small.webp" alt="Description">
</picture>

<!-- CSS optimization -->
<link rel="stylesheet" href="critical.css">
<link rel="stylesheet" href="non-critical.css" media="print" onload="this.media='all'">

<!-- JavaScript optimization -->
<script src="app.js" defer></script>
<script src="analytics.js" async></script>
```

### 4.4 Core Web Vitals

```
┌─────────────────────────────────────────────────────────────────┐
│                    Core Web Vitals Metrics                       │
│                                                                 │
│   LCP (Largest Contentful Paint):                              │
│   - Largest content element load time                          │
│   - Target: Under 2.5 seconds                                  │
│   - Improve: Image optimization, server response time          │
│                                                                 │
│   FID (First Input Delay) → INP (Interaction to Next Paint):   │
│   - First interaction delay → Overall interaction responsiveness│
│   - Target: Under 100ms → Under 200ms                          │
│   - Improve: JavaScript optimization, minimize main thread     │
│            blocking                                             │
│                                                                 │
│   CLS (Cumulative Layout Shift):                               │
│   - Cumulative layout change score                             │
│   - Target: Under 0.1                                          │
│   - Improve: Specify image/ad sizes, optimize font loading    │
└─────────────────────────────────────────────────────────────────┘
```

### 4.5 Mobile Optimization

```html
<!-- Mobile-friendly settings -->
<meta name="viewport" content="width=device-width, initial-scale=1">

<!-- Touch target size (minimum 48x48px) -->
<style>
button, a {
  min-height: 48px;
  min-width: 48px;
  padding: 12px 16px;
}
</style>

<!-- Readability (minimum 16px font) -->
<style>
body {
  font-size: 16px;
  line-height: 1.5;
}
</style>
```

---

## 5. Content SEO

### 5.1 Keyword Optimization

```html
<!-- Include main keywords in title -->
<title>Web Development Basics | HTML, CSS, JavaScript Tutorial</title>

<!-- One H1 per page, include keywords -->
<h1>Web Development Basics Guide: Getting Started with HTML, CSS, JavaScript</h1>

<!-- Natural keyword placement -->
<p>
  To start <strong>web development</strong>, you need to understand the basics
  of HTML, CSS, and JavaScript. This <em>web development tutorial</em>...
</p>
```

### 5.2 Semantic Markup

```html
<article>
  <header>
    <h1>Web Development Basics Guide</h1>
    <p>
      <time datetime="2024-03-20">March 20, 2024</time> |
      <a href="/author/john">Author Name</a>
    </p>
  </header>

  <section>
    <h2>HTML Basics</h2>
    <p>HTML defines the structure of web pages...</p>
  </section>

  <section>
    <h2>CSS Styling</h2>
    <p>CSS handles visual presentation...</p>
  </section>

  <aside>
    <h3>Related Posts</h3>
    <ul>
      <li><a href="/css-layout">Master CSS Layout</a></li>
      <li><a href="/javascript-basics">JavaScript Introduction</a></li>
    </ul>
  </aside>

  <footer>
    <p>Tags: <a href="/tag/html">HTML</a>, <a href="/tag/css">CSS</a></p>
  </footer>
</article>
```

### 5.3 Link Optimization

```html
<!-- Internal links (clear anchor text) -->
<a href="/css-tutorial">View CSS Tutorial</a>
<!-- Avoid: <a href="/css-tutorial">Click here</a> -->

<!-- External links -->
<a href="https://developer.mozilla.org"
   target="_blank"
   rel="noopener noreferrer">
  MDN Web Docs
</a>

<!-- nofollow (untrusted links) -->
<a href="https://external-site.com" rel="nofollow">External Site</a>

<!-- Breadcrumbs -->
<nav aria-label="Breadcrumb">
  <ol>
    <li><a href="/">Home</a></li>
    <li><a href="/tutorials">Tutorials</a></li>
    <li aria-current="page">Web Development Basics</li>
  </ol>
</nav>
```

### 5.4 URL Structure

```
Good URL Structure:
✓ https://example.com/tutorials/web-development-basics
✓ https://example.com/products/electronics/smartphones

URLs to Avoid:
✗ https://example.com/page?id=123&cat=5
✗ https://example.com/p/a/b/c/d/e/article
✗ https://example.com/웹개발 (Korean URLs get encoded)

Rules:
- Use lowercase
- Separate words with hyphens (-)
- Be concise and descriptive
- Reflect hierarchy
- Remove unnecessary parameters
```

---

## 6. Measurement and Tools

### 6.1 Google Search Console

```
Main Features:
- Search performance analysis (clicks, impressions, CTR, ranking)
- Index status check
- Sitemap submission
- Crawling error check
- Core Web Vitals report
- Mobile usability report

Setup:
1. Access Google Search Console
2. Verify site ownership (HTML tag, DNS, file upload)
3. Submit sitemap
```

### 6.2 Google Analytics 4

```html
<!-- GA4 installation -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-XXXXXXXXXX"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-XXXXXXXXXX');
</script>

<!-- Event tracking -->
<script>
  // Track button click
  document.querySelector('.cta-button').addEventListener('click', () => {
    gtag('event', 'cta_click', {
      'event_category': 'engagement',
      'event_label': 'signup_button'
    });
  });
</script>
```

### 6.3 SEO Tools

```
Free Tools:
- Google Search Console: Search performance, index status
- Google PageSpeed Insights: Page speed
- Lighthouse: Comprehensive website quality measurement
- Screaming Frog (free version): Crawling, site analysis
- Google Rich Results Test: Test structured data

Paid Tools:
- Ahrefs: Backlink analysis, keyword research
- SEMrush: Competitor analysis, keyword tracking
- Moz Pro: Domain authority, SEO analysis
```

### 6.4 Checklist

```
┌─────────────────────────────────────────────────────────────────┐
│                   SEO Checklist                                  │
│                                                                 │
│ Basic Setup:                                                    │
│ □ Unique and descriptive title tags                            │
│ □ Compelling meta description                                  │
│ □ Appropriate canonical URL                                    │
│ □ robots.txt configured                                        │
│ □ sitemap.xml created and submitted                           │
│ □ HTTPS enabled                                                │
│                                                                 │
│ Content:                                                        │
│ □ H1 tag (one per page)                                        │
│ □ Logical heading structure (H1→H2→H3)                         │
│ □ Image alt text                                               │
│ □ Internal link structure                                      │
│ □ Mobile-friendly content                                      │
│                                                                 │
│ Technical:                                                      │
│ □ Core Web Vitals passing                                      │
│ □ Mobile responsive                                            │
│ □ Structured data                                              │
│ □ 404 page handling                                            │
│ □ Redirect optimization                                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Practice Problems

### Exercise 1: Write Meta Tags
Write necessary meta tags for a blog post.

```html
<!-- Example answer -->
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">

  <title>Complete React Hooks Guide - From useState to Custom Hooks | Blog Name</title>
  <meta name="description" content="Learn everything about React Hooks.
    From useState, useEffect, useContext to custom hooks explained with examples.">

  <link rel="canonical" href="https://blog.example.com/react-hooks-guide">

  <!-- Open Graph -->
  <meta property="og:type" content="article">
  <meta property="og:title" content="Complete React Hooks Guide">
  <meta property="og:description" content="Everything about React Hooks with examples">
  <meta property="og:image" content="https://blog.example.com/images/react-hooks-og.jpg">
  <meta property="og:url" content="https://blog.example.com/react-hooks-guide">

  <!-- Twitter Card -->
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:title" content="Complete React Hooks Guide">
  <meta name="twitter:description" content="Everything about React Hooks with examples">
  <meta name="twitter:image" content="https://blog.example.com/images/react-hooks-twitter.jpg">
</head>
```

### Exercise 2: Write Structured Data
Write structured data for a recipe page.

```html
<!-- Example answer -->
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "Recipe",
  "name": "Spaghetti Carbonara",
  "image": "https://example.com/carbonara.jpg",
  "author": {
    "@type": "Person",
    "name": "Chef Name"
  },
  "datePublished": "2024-03-15",
  "description": "Authentic Italian carbonara recipe",
  "prepTime": "PT10M",
  "cookTime": "PT20M",
  "totalTime": "PT30M",
  "recipeYield": "4 servings",
  "recipeIngredient": [
    "400g spaghetti",
    "200g pancetta",
    "4 eggs",
    "100g Parmesan cheese"
  ],
  "recipeInstructions": [
    {
      "@type": "HowToStep",
      "text": "Cook spaghetti according to package directions"
    },
    {
      "@type": "HowToStep",
      "text": "Fry pancetta until crispy"
    }
  ],
  "aggregateRating": {
    "@type": "AggregateRating",
    "ratingValue": "4.8",
    "ratingCount": "156"
  }
}
</script>
```

### Exercise 3: Write robots.txt
Write a robots.txt meeting the following requirements.
- Block admin pages (/admin/)
- Block API endpoints (/api/)
- Block search result pages (/search?*)
- Allow everything else

```
# Example answer
User-agent: *
Allow: /
Disallow: /admin/
Disallow: /api/
Disallow: /search

Sitemap: https://example.com/sitemap.xml
```

---

## References
- [Google Search Central](https://developers.google.com/search)
- [Schema.org](https://schema.org/)
- [Web.dev SEO Guide](https://web.dev/learn/seo/)
- [Moz Beginner's Guide to SEO](https://moz.com/beginners-guide-to-seo)

---

**Previous**: [Web Accessibility](./11_Web_Accessibility.md) | **Next**: [Build Tools & Development Environment](./13_Build_Tools_Environment.md)
