mkdir -p data
cd data/

# Retrieve data directly from Stanford data source
if ! wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
then
    echo "Trying curl..."
    if ! curl http://cs231n.stanford.edu/tiny-imagenet-200.zip -o tiny-imagenet-200.zip
    then 
        echo "Error: Curl and wget failed!"
        exit 0
    fi
fi
  
# Unzip raw zip file
unzip -qq 'tiny-imagenet-200.zip'
rm tiny-imagenet-200.zip