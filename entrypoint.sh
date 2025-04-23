#!/bin/bash
# Wait for MongoDB to be ready (optional)
sleep 5

# Run migrations and collect static files
python manage.py collectstatic --noinput

# Run the server
exec "$@"
