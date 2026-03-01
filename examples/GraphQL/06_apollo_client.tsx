/**
 * GraphQL Client — Apollo Client 3 with React
 * Demonstrates: useQuery, useMutation, cache, optimistic updates.
 *
 * Setup: npm install @apollo/client graphql
 */

import React from 'react';
import {
  ApolloClient,
  InMemoryCache,
  ApolloProvider,
  gql,
  useQuery,
  useMutation,
  // useSubscription,
} from '@apollo/client';

// --- Apollo Client Setup ---

const client = new ApolloClient({
  uri: 'http://localhost:4000/graphql',
  cache: new InMemoryCache({
    typePolicies: {
      Query: {
        fields: {
          posts: {
            // Merge function for pagination
            merge(existing = [], incoming: any[]) {
              return [...existing, ...incoming];
            },
          },
        },
      },
    },
  }),
});

// --- GraphQL Operations ---

const GET_POSTS = gql`
  query GetPosts($status: Status) {
    posts(status: $status) {
      id
      title
      content
      status
      author {
        id
        name
      }
    }
  }
`;

const GET_POST = gql`
  query GetPost($id: ID!) {
    post(id: $id) {
      id
      title
      content
      status
      author {
        id
        name
        email
      }
    }
  }
`;

const CREATE_POST = gql`
  mutation CreatePost($input: CreatePostInput!) {
    createPost(input: $input) {
      id
      title
      content
      status
      author {
        id
        name
      }
    }
  }
`;

const DELETE_POST = gql`
  mutation DeletePost($id: ID!) {
    deletePost(id: $id)
  }
`;

// --- Components ---

function PostList() {
  const { loading, error, data, refetch } = useQuery(GET_POSTS, {
    variables: { status: 'PUBLISHED' },
    // fetchPolicy: 'cache-and-network', // Show cache then update
    // pollInterval: 5000,               // Poll every 5 seconds
  });

  const [deletePost] = useMutation(DELETE_POST, {
    // Update cache after deletion
    update(cache, { data: { deletePost: success } }, { variables }) {
      if (success && variables) {
        cache.modify({
          fields: {
            posts(existingPosts = [], { readField }) {
              return existingPosts.filter(
                (postRef: any) => readField('id', postRef) !== variables.id
              );
            },
          },
        });
      }
    },
  });

  if (loading) return <p>Loading...</p>;
  if (error) return <p>Error: {error.message}</p>;

  return (
    <div>
      <h1>Posts</h1>
      <button onClick={() => refetch()}>Refresh</button>

      {data.posts.map((post: any) => (
        <div key={post.id} style={{ border: '1px solid #ccc', padding: '12px', margin: '8px 0' }}>
          <h3>{post.title}</h3>
          <p>{post.content}</p>
          <small>By {post.author.name}</small>
          <button onClick={() => deletePost({ variables: { id: post.id } })}>
            Delete
          </button>
        </div>
      ))}
    </div>
  );
}

function CreatePostForm() {
  const [title, setTitle] = React.useState('');
  const [content, setContent] = React.useState('');

  const [createPost, { loading }] = useMutation(CREATE_POST, {
    // Optimistic response: update UI immediately before server responds
    optimisticResponse: {
      createPost: {
        __typename: 'Post',
        id: `temp-${Date.now()}`,
        title,
        content,
        status: 'DRAFT',
        author: {
          __typename: 'Author',
          id: '1',
          name: 'Current User',
        },
      },
    },
    // Update cache with new post
    update(cache, { data: { createPost: newPost } }) {
      cache.modify({
        fields: {
          posts(existingPosts = []) {
            const newPostRef = cache.writeFragment({
              data: newPost,
              fragment: gql`
                fragment NewPost on Post {
                  id
                  title
                  content
                  status
                  author { id name }
                }
              `,
            });
            return [...existingPosts, newPostRef];
          },
        },
      });
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    createPost({
      variables: { input: { title, content, authorId: '1' } },
    });
    setTitle('');
    setContent('');
  };

  return (
    <form onSubmit={handleSubmit}>
      <h2>Create Post</h2>
      <input
        value={title}
        onChange={(e) => setTitle(e.target.value)}
        placeholder="Title"
        required
      />
      <textarea
        value={content}
        onChange={(e) => setContent(e.target.value)}
        placeholder="Content"
      />
      <button type="submit" disabled={loading}>
        {loading ? 'Creating...' : 'Create'}
      </button>
    </form>
  );
}

// --- App ---

function App() {
  return (
    <ApolloProvider client={client}>
      <div style={{ maxWidth: 600, margin: '0 auto', padding: 20 }}>
        <CreatePostForm />
        <PostList />
      </div>
    </ApolloProvider>
  );
}

export default App;
