const { default: axios } = require("axios");

module.exports = function (app) {
  // console.log("api called");
  app.get("/api/ss", function (req, res) {
    console.log("api called");
    res.send({ message: "Super secret code is ABC123" });
  });

  app.get("/api/sym", function (req, res) {
    console.log("api called");
    axios
      .get(`http://localhost:5002/symsroutine`)
      .then((response) => {
        // console.log(response);
        console.log(response.data);
      })
      .catch((err) => {
        console.log("err" + err);
      });
    // res.send({ message: "Super secret code is ABC123" });
  });

  app.get("/api/xray", function (req, res) {
    console.log("api called");
    axios
      .get(`http://localhost:5002/xray`)
      .then((response) => {
        // console.log(response);
        console.log(response.data);
      })
      .catch((err) => {
        console.log("err" + err);
      });
    // res.send({ message: "Super secret code is ABC123" });
  });

  app.post("/api/sym", function (req, res) {
    console.log("api called");
    console.log(req.body);
    axios
      .post(`http://localhost:5002/symsroutine`, req.body)
      .then((response) => {
        // console.log(response);
        // console.log(typeof response.data);//--> object
        // console.log(typeof str(response.data));
        // console.log(JSON.parse(str(response.data)));
        console.log(response.data["y_pred-algo_details"]);
        console.log(response.data["y_pred-votes"]);
      })
      .catch((err) => {
        console.log("err" + err);
      });
    // res.send({ message: "Super secret code is ABC123" });
  });

  app.post("/api/xray", function (req, res) {
    console.log("api called");
    console.log(req.body);
    b64url = req.body.b64url;
    axios
      .post(`http://localhost:5002/xray`, {
        b64url: b64url.substring(b64url.indexOf(",") + 1),
      })
      .then((response) => {
        // console.log(response);
        console.log(response.data);
      })
      .catch((err) => {
        console.log("err" + err);
      });
    // res.send({ message: "Super secret code is ABC123" });
  });
};
