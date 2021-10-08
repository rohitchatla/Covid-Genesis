import React, { Component } from "react";
import ReactDOM from "react-dom";
import JupyterViewer from "react-jupyter-notebook";
import nb_test from "./nb_test.json"; // You need to read the .ipynb file into a JSON Object.
import axios from "../services/Axios";

class jnb extends Component {
  constructor(props) {
    super(props);
    this.state = {
      work: [],
    };
  }
  async componentDidMount() {
    console.log("hi");
    // await axios
    //   .get(`/timeseries`) // axios returns a promise
    //   .then((response) => {
    //     console.log(response.data);
    //   })
    //   .catch(({ response }) => {
    //     console.log(response);
    //   });
  }

  render() {
    return (
      <>
        <React.StrictMode>
          <JupyterViewer rawIpynb={nb_test} />
        </React.StrictMode>
        , document.getElementById("root")
      </>
    );
  }
}

export default jnb;
