/**
 * Exercise: GraphQL Resolvers
 * Practice writing resolvers with context, DataLoader, and error handling.
 *
 * Run: npm install @apollo/server graphql dataloader
 */

// Exercise 1: Implement Resolvers
// Given this schema, implement the resolvers:

const typeDefs = `#graphql
  type Author {
    id: ID!
    name: String!
    books: [Book!]!
    bookCount: Int!
  }

  type Book {
    id: ID!
    title: String!
    genre: String!
    author: Author!
    reviews: [Review!]!
    averageRating: Float
  }

  type Review {
    id: ID!
    rating: Int!
    comment: String
    reviewer: String!
  }

  type Query {
    books(genre: String): [Book!]!
    book(id: ID!): Book
    authors: [Author!]!
    author(id: ID!): Author
    topRated(limit: Int = 5): [Book!]!
  }

  type Mutation {
    addBook(title: String!, genre: String!, authorId: ID!): Book!
    addReview(bookId: ID!, rating: Int!, comment: String, reviewer: String!): Review!
  }
`;

// Mock data
const authors = [
  { id: '1', name: 'J.K. Rowling' },
  { id: '2', name: 'George Orwell' },
  { id: '3', name: 'Tolkien' },
];

const books = [
  { id: '1', title: 'Harry Potter', genre: 'Fantasy', authorId: '1' },
  { id: '2', title: '1984', genre: 'Dystopian', authorId: '2' },
  { id: '3', title: 'The Hobbit', genre: 'Fantasy', authorId: '3' },
];

const reviews = [
  { id: '1', bookId: '1', rating: 5, comment: 'Amazing!', reviewer: 'Alice' },
  { id: '2', bookId: '1', rating: 4, comment: 'Great read', reviewer: 'Bob' },
  { id: '3', bookId: '2', rating: 5, comment: 'Classic', reviewer: 'Charlie' },
];

// TODO: Implement resolvers
const resolvers = {
  Query: {
    // TODO: books (with optional genre filter)
    // TODO: book (by id)
    // TODO: authors
    // TODO: author (by id)
    // TODO: topRated (sorted by average rating, limited)
  },

  Mutation: {
    // TODO: addBook (validate authorId exists)
    // TODO: addReview (validate bookId exists, rating 1-5)
  },

  // TODO: Field resolvers for Author, Book
  Author: {
    // TODO: books — return books by this author
    // TODO: bookCount — return count of books
  },

  Book: {
    // TODO: author — resolve author from authorId
    // TODO: reviews — return reviews for this book
    // TODO: averageRating — compute average from reviews
  },
};


// Exercise 2: DataLoader
// Create DataLoaders to batch-load:
// - Authors by ID (for Book.author)
// - Reviews by book ID (for Book.reviews)
// Verify that N+1 is eliminated.

// TODO: Implement createLoaders function


// Exercise 3: Error Handling
// Modify resolvers to throw proper GraphQL errors:
// - addBook with non-existent authorId → VALIDATION_ERROR
// - addReview with invalid rating → BAD_USER_INPUT
// - book(id) not found → NOT_FOUND
// Use GraphQLError with extensions.code

// TODO: Add error handling


// Exercise 4: Pagination
// Add cursor-based pagination to the books query:
// books(first: Int, after: String, genre: String): BookConnection!
// type BookConnection { edges: [BookEdge!]!, pageInfo: PageInfo! }
// type BookEdge { node: Book!, cursor: String! }
// type PageInfo { hasNextPage: Boolean!, endCursor: String }

// TODO: Implement cursor pagination resolver


module.exports = { typeDefs, resolvers };
