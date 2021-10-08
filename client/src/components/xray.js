import React, { Component } from "react";
import { Link } from "react-router-dom";
import axios from "../services/Axios";
// components

class xray extends Component {
  constructor(props) {
    super(props);
    this.state = {
      image: "",
      b64url: "",
    };
  }
  async componentDidMount() {
    // console.log("hi");
    // await axios
    //   .get(`/xray`) // axios returns a promise
    //   .then((response) => {
    //     console.log(response.data);
    //   })
    //   .catch(({ response }) => {
    //     console.log(response);
    //   });
  }

  handleSubmit() {
    const { image, b64url } = this.state;
    // const uid = localStorage.getItem("uid");

    let obj = {
      // image: JSON.stringify(image),
      b64url, //.substring(b64url.indexOf(",") + 1),
    };

    // const dataform = new FormData();

    // dataform.append("image", image);
    // dataform.append("b64url", b64url);

    // console.log(b64url);

    axios
      .post("/xray", obj, {
        //dataform
        headers: {
          //"Content-Type": "multipart/form-data",
          //"x-access-token": `${localStorage.getItem("token")}`,
        },
      }) // axios returns a promise
      .then((response) => {
        // window.open("/xray", "_self");
        alert("Submitted");
      })
      .catch(({ response }) => {});
  }

  async getBase64(file, cb) {
    let reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = function () {
      cb(reader.result);
    };
    reader.onerror = function (error) {
      console.log("Error: ", error);
    };
  }

  render() {
    const { image, b64url } = this.state;
    return (
      <>
        <div>
          {/* <h2>Hi</h2> */}
          <h2 className="mb-5">Upload Lungs X Ray</h2>
          {/* <form onSubmit={this.handleSubmit()}> */}
          <input
            accept="image/*"
            onChange={(e) => {
              let idCardBase64 = "";
              this.getBase64(e.target.files[0], (result) => {
                idCardBase64 = result;
                this.setState({ b64url: result });
              });

              this.setState({ image: e.target.files[0] });
            }}
            type="file"
          />
          <button
            action="submit"
            className="btn btn-primary"
            onClick={() => this.handleSubmit()}
          >
            Scan
          </button>
          {/* </form> */}
        </div>
      </>
    );
  }
}

export default xray;
