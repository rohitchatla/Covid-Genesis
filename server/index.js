// Main starting point of the application
const express = require("express");
const http = require("http");
const bodyParser = require("body-parser");
const app = express();
const router = require("./router");
const mongoose = require("mongoose");
const axios = require("axios");

// import {PythonShell} from 'python-shell';
const { PythonShell } = require("python-shell");
const path = require("path");
// const cors = require('cors');  // we don't need it anymore, because we use proxy server instead

// DB Setup (connect mongoose and instance of mongodb)
// mongoose.connect(
//   process.env.MONGO_DB_URL ||
//     "mongodb+srv://funnyguys:funnyguys@cluster0.dk88w.mongodb.net/inout?retryWrites=true&w=majority",
//   {
//     useNewUrlParser: true,
//     useCreateIndex: true,
//     useUnifiedTopology: true,
//     useFindAndModify: false,
//   }
// );
const cors = require("cors");
app.use(cors());

//app.use(bodyParser.json({ type: "*/*" })); // middleware for helping parse incoming HTTP requests
app.use(cors()); // middleware for circumventing cors error
app.use(bodyParser.json());

app.use(bodyParser());

app.use(bodyParser.urlencoded({ extended: false }));
app.use("/assets", express.static("assets"));

app.get("/api/timeseries", function (req, res) {
  let options = {
    mode: "text",
    pythonPath:
      "C:\\Users\\rohit\\AppData\\Local\\Programs\\Python\\Python37\\python.exe",
    pythonOptions: ["-u"], // get print results in real-time
    scriptPath: path.join(
      __dirname,
      "..",
      "covid-19-LSTM-Analysis-ML-DeepLearning-master"
    ),
    encoding: "utf-8",
    stderrParser: (...errors) => console.log("PYERR:", errors.join("\n")),
    //args: ["value1", "value2", "value3"],
  };
  PythonShell.run(
    "../covid-19-LSTM-Analysis-ML-DeepLearning-master/routine.py",
    options,
    function (err, results) {
      //null
      if (err) throw err;
      console.log("finished");
      console.log("results: %j", results);
    }
  );
});
// Router Setup
router(app);

// Server Setup
const port = process.env.PORT || 5000;
const server = http.createServer(app);
server.listen(port);
console.log("Server listening on: ", port);

// axios
//   .get(`https://randomuser.me/api`)
//   .then((response) => {
//     console.log(response.data.results);
//   })
//   .catch(({ response }) => {});

/**  Sockets routines **/
