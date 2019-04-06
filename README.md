**Chobot project autokeras Neural Architecture Search model with API**  
Neural arcitecture search implementation with autokeras for chobot project.

**Push to local Docker registry**  
```
docker build -t chobot_autokeras_nas_image:latest . 
docker tag chobot_autokeras_nas_image:latest localhost:5000/chobot_autokeras_nas_image:latest
docker push localhost:5000/chobot_autokeras_nas_image:latest
```

**Push to Docker hub**  
```
docker build -t chobot_autokeras_nas_image:latest . 
docker tag chobot_autokeras_nas_image:latest stejsky/chobot_autokeras_nas_image:latest
docker push stejsky/chobot_autokeras_nas_image:latest
```

