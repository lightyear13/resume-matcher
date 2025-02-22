!/bin/sh

echo ""
echo "Loading azd .env file from current environment"
echo ""

# while IFS='=' read -r key value; do
#     value=$(echo "$value" | sed 's/^"//' | sed 's/"$//')
#     export "$key=$value"
# done <<EOF
# $(azd env get-values)
# EOF

if [ $? -ne 0 ]; then
    echo "Failed to load environment variables from azd environment"
    exit $?
fi

echo 'Creating python virtual environment "backend/holaa"'
python3 -m venv backend/holaa

echo ""
echo "Restoring backend python packages"
echo ""

cd backend
./holaa/bin/python -m pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Failed to restore backend python packages"
    exit $?
fi

echo ""
echo "Restoring frontend npm packages"
echo ""

cd ../resume-matcher-ui
npm install
if [ $? -ne 0 ]; then
    echo "Failed to restore frontend npm packages"
    exit $?
fi

echo ""
echo "Building frontend"
echo ""

# npm run build
# if [ $? -ne 0 ]; then
#     echo "Failed to build frontend"
#     exit $?
# fi

echo ""
echo "Starting backend"
echo ""

cd ../backend
xdg-open http://127.0.0.1:5000
./holaa/bin/python -m quart --app main:app run --port 5000 --reload
if [ $? -ne 0 ]; then
    echo "Failed to start backend"
    exit $?
fi