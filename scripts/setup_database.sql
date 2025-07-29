-- Sample Database Setup for CrewAI SQL Agent
-- This script creates a sample e-commerce database with users, products, orders, and order_items tables

-- Create database (run this separately if needed)
-- CREATE DATABASE sample_db;

-- Connect to the database
-- \c sample_db;

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'active'
);

-- Create products table
CREATE TABLE IF NOT EXISTS products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    price DECIMAL(10,2) NOT NULL,
    category VARCHAR(50),
    stock_quantity INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create orders table
CREATE TABLE IF NOT EXISTS orders (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    status VARCHAR(20) DEFAULT 'pending',
    total_amount DECIMAL(10,2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create order_items table
CREATE TABLE IF NOT EXISTS order_items (
    id SERIAL PRIMARY KEY,
    order_id INTEGER REFERENCES orders(id),
    product_id INTEGER REFERENCES products(id),
    quantity INTEGER NOT NULL,
    unit_price DECIMAL(10,2) NOT NULL,
    total_price DECIMAL(10,2) NOT NULL
);

-- Insert sample data

-- Insert users
INSERT INTO users (name, email, status) VALUES
('John Doe', 'john.doe@example.com', 'active'),
('Jane Smith', 'jane.smith@example.com', 'active'),
('Bob Johnson', 'bob.johnson@example.com', 'inactive'),
('Alice Brown', 'alice.brown@example.com', 'active'),
('Charlie Wilson', 'charlie.wilson@example.com', 'active'),
('Diana Davis', 'diana.davis@example.com', 'active'),
('Edward Miller', 'edward.miller@example.com', 'inactive'),
('Fiona Garcia', 'fiona.garcia@example.com', 'active'),
('George Martinez', 'george.martinez@example.com', 'active'),
('Helen Rodriguez', 'helen.rodriguez@example.com', 'active');

-- Insert products
INSERT INTO products (name, description, price, category, stock_quantity) VALUES
('Laptop Pro', 'High-performance laptop for professionals', 1299.99, 'Electronics', 50),
('Smartphone X', 'Latest smartphone with advanced features', 799.99, 'Electronics', 100),
('Wireless Headphones', 'Noise-cancelling wireless headphones', 199.99, 'Electronics', 75),
('Coffee Maker', 'Automatic coffee maker with timer', 89.99, 'Home & Kitchen', 30),
('Running Shoes', 'Comfortable running shoes for athletes', 129.99, 'Sports', 60),
('Backpack', 'Durable backpack for daily use', 59.99, 'Fashion', 40),
('Tablet', '10-inch tablet with high-resolution display', 399.99, 'Electronics', 25),
('Blender', 'High-speed blender for smoothies', 79.99, 'Home & Kitchen', 35),
('Yoga Mat', 'Non-slip yoga mat for fitness', 29.99, 'Sports', 80),
('Watch', 'Elegant watch with leather strap', 149.99, 'Fashion', 45);

-- Insert orders
INSERT INTO orders (user_id, status, total_amount) VALUES
(1, 'completed', 1299.99),
(2, 'completed', 999.98),
(3, 'pending', 299.97),
(4, 'completed', 179.98),
(5, 'shipped', 259.98),
(6, 'completed', 399.99),
(7, 'cancelled', 159.98),
(8, 'completed', 89.99),
(9, 'pending', 449.97),
(10, 'completed', 179.97);

-- Insert order items
INSERT INTO order_items (order_id, product_id, quantity, unit_price, total_price) VALUES
(1, 1, 1, 1299.99, 1299.99),
(2, 2, 1, 799.99, 799.99),
(2, 3, 1, 199.99, 199.99),
(3, 4, 1, 89.99, 89.99),
(3, 5, 1, 129.99, 129.99),
(3, 6, 1, 59.99, 59.99),
(4, 7, 1, 399.99, 399.99),
(4, 8, 1, 79.99, 79.99),
(5, 9, 1, 29.99, 29.99),
(5, 10, 1, 149.99, 149.99),
(6, 1, 1, 1299.99, 1299.99),
(7, 2, 1, 799.99, 799.99),
(7, 3, 1, 199.99, 199.99),
(8, 4, 1, 89.99, 89.99),
(9, 5, 1, 129.99, 129.99),
(9, 6, 1, 59.99, 59.99),
(9, 7, 1, 399.99, 399.99),
(10, 8, 1, 79.99, 79.99),
(10, 9, 1, 29.99, 29.99),
(10, 10, 1, 149.99, 149.99);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_status ON users(status);
CREATE INDEX IF NOT EXISTS idx_products_category ON products(category);
CREATE INDEX IF NOT EXISTS idx_orders_user_id ON orders(user_id);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_order_items_order_id ON order_items(order_id);
CREATE INDEX IF NOT EXISTS idx_order_items_product_id ON order_items(product_id);

-- Display table information
SELECT 'Database setup completed successfully!' as status;

-- Show table counts
SELECT 'users' as table_name, COUNT(*) as record_count FROM users
UNION ALL
SELECT 'products', COUNT(*) FROM products
UNION ALL
SELECT 'orders', COUNT(*) FROM orders
UNION ALL
SELECT 'order_items', COUNT(*) FROM order_items; 