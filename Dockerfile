FROM node:20
WORKDIR /usr/src/app
ENV PORT=8000
COPY . . 
RUN npm install
COPY package*.json ./
CMD [ "npm", "run", "start" ]