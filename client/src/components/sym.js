import React, { Component } from "react";
// import { Link } from "react-router-dom";
import axios from "../services/Axios";

// import * as React from 'react';
import Avatar from "@mui/material/Avatar";
import Button from "@mui/material/Button";
import CssBaseline from "@mui/material/CssBaseline";
import TextField from "@mui/material/TextField";
import FormControlLabel from "@mui/material/FormControlLabel";
import Checkbox from "@mui/material/Checkbox";
import Link from "@mui/material/Link";
import Grid from "@mui/material/Grid";
import Box from "@mui/material/Box";
import LockOutlinedIcon from "@mui/icons-material/LockOutlined";
import Typography from "@mui/material/Typography";
import Container from "@mui/material/Container";
import { createTheme, ThemeProvider } from "@mui/material/styles";

import LocalHospital from "@mui/icons-material/LocalHospital";

// components

function Copyright(props) {
  return (
    <Typography
      variant="body2"
      color="text.secondary"
      align="center"
      {...props}
    >
      {"Copyright © "}
      <Link color="inherit" href="#">
        Covid-19 Genesis
      </Link>{" "}
      {new Date().getFullYear()}
      {"."}
    </Typography>
  );
}

class sym extends Component {
  constructor(props) {
    super(props);
    this.state = {
      work: [],
    };
  }
  async componentDidMount() {
    // console.log("hi");
    // await axios
    //   .get(`/sym`) // axios returns a promise
    //   .then((response) => {
    //     console.log(response.data);
    //   })
    //   .catch(({ response }) => {
    //     console.log(response);
    //   });
  }

  handleSubmit = async (event) => {
    event.preventDefault();
    const data = new FormData(event.currentTarget);
    // eslint-disable-next-line no-console
    if (
      data.get("Breathing_Problem") == "" ||
      data.get("Fever") == "" ||
      data.get("Dry_Cough") == "" ||
      data.get("Sore_throat") == "" ||
      data.get("Hyper_Tension") == "" ||
      data.get("Abroad_travel") == "" ||
      data.get("Contact_with_COVID_Patient") == "" ||
      data.get("Attended_Large_Gathering") == "" ||
      data.get("Visited_Public_Exposed_Places") == "" ||
      data.get("Family_working_in_Public_Exposed_Places") == ""
    ) {
      alert("Fill up all details");
      return;
    }
    console.log({
      Breathing_Problem: data.get("Breathing_Problem"),
      Fever: data.get("Fever"),

      Dry_Cough: data.get("Dry_Cough"),
      Sore_throat: data.get("Sore_throat"),
      Hyper_Tension: data.get("Hyper_Tension"),
      Abroad_travel: data.get("Abroad_travel"),
      Contact_with_COVID_Patient: data.get("Contact_with_COVID_Patient"),
      Attended_Large_Gathering: data.get("Attended_Large_Gathering"),
      Visited_Public_Exposed_Places: data.get("Visited_Public_Exposed_Places"),

      Family_working_in_Public_Exposed_Places: data.get(
        "Family_working_in_Public_Exposed_Places"
      ),
    });
    let dataObj = {};
    dataObj = {
      Breathing_Problem: data.get("Breathing_Problem"),
      Fever: data.get("Fever"),

      Dry_Cough: data.get("Dry_Cough"),
      Sore_throat: data.get("Sore_throat"),
      Hyper_Tension: data.get("Hyper_Tension"),
      Abroad_travel: data.get("Abroad_travel"),
      Contact_with_COVID_Patient: data.get("Contact_with_COVID_Patient"),
      Attended_Large_Gathering: data.get("Attended_Large_Gathering"),
      Visited_Public_Exposed_Places: data.get("Visited_Public_Exposed_Places"),

      Family_working_in_Public_Exposed_Places: data.get(
        "Family_working_in_Public_Exposed_Places"
      ),
    };

    await axios
      .post(`/sym`, dataObj) // axios returns a promise
      .then((response) => {
        console.log(response.data);
      })
      .catch(({ response }) => {
        console.log(response);
      });
  };

  render() {
    const theme = createTheme();
    return (
      // <>
      //   <div>
      //     <h2>Hi</h2>
      //   </div>
      // </>
      <ThemeProvider theme={theme}>
        <Container component="main" maxWidth="xs">
          <CssBaseline />
          <Box
            sx={{
              marginTop: 8,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
            }}
          >
            <Avatar sx={{ m: 1, bgcolor: "secondary.main" }}>
              <LocalHospital />
            </Avatar>
            <Typography component="h1" variant="h5">
              Track for Covid-19
            </Typography>
            <Box
              component="form"
              noValidate
              onSubmit={this.handleSubmit}
              sx={{ mt: 3 }}
            >
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6}>
                  <TextField
                    autoComplete="Breathing_Problem"
                    name="Breathing_Problem"
                    required
                    fullWidth
                    id="Breathing_Problem"
                    label="Breathing_Problem"
                    autoFocus
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    required
                    fullWidth
                    id="Fever"
                    label="Fever"
                    name="Fever"
                    autoComplete="Fever"
                  />
                </Grid>

                <Grid item xs={12} sm={6}>
                  <TextField
                    autoComplete="Dry_Cough"
                    name="Dry_Cough"
                    required
                    fullWidth
                    id="Dry_Cough"
                    label="Dry_Cough"
                    autoFocus
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    required
                    fullWidth
                    id="Sore_throat"
                    label="Sore_throat"
                    name="Sore_throat"
                    autoComplete="Sore_throat"
                  />
                </Grid>

                <Grid item xs={12} sm={6}>
                  <TextField
                    autoComplete="Hyper_Tension"
                    name="Hyper_Tension"
                    required
                    fullWidth
                    id="Hyper_Tension"
                    label="Hyper_Tension"
                    autoFocus
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    required
                    fullWidth
                    id="Abroad_travel"
                    label="Abroad_travel"
                    name="Abroad_travel"
                    autoComplete="Abroad_travel"
                  />
                </Grid>

                <Grid item xs={12} sm={6}>
                  <TextField
                    autoComplete="Contact_with_COVID_Patient"
                    name="Contact_with_COVID_Patient"
                    required
                    fullWidth
                    id="Contact_with_COVID_Patient"
                    label="Contact_with_COVID_Patient"
                    autoFocus
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    required
                    fullWidth
                    id="Attended_Large_Gathering"
                    label="Attended_Large_Gathering"
                    name="Attended_Large_Gathering"
                    autoComplete="Attended_Large_Gathering"
                  />
                </Grid>

                <Grid item xs={12}>
                  <TextField
                    required
                    fullWidth
                    id="Visited_Public_Exposed_Places"
                    label="Visited_Public_Exposed_Places"
                    name="Visited_Public_Exposed_Places"
                    autoComplete="Visited_Public_Exposed_Places"
                  />
                </Grid>
                <Grid item xs={12}>
                  <TextField
                    required
                    fullWidth
                    name="Family_working_in_Public_Exposed_Places"
                    label="Family_working_in_Public_Exposed_Places"
                    type="Family_working_in_Public_Exposed_Places"
                    id="Family_working_in_Public_Exposed_Places"
                    autoComplete="Family_working_in_Public_Exposed_Places"
                  />
                </Grid>
                {/* <Grid item xs={12}>
                  <FormControlLabel
                    control={
                      <Checkbox value="allowExtraEmails" color="primary" />
                    }
                    label="I want to receive inspiration, marketing promotions and updates via email."
                  />
                </Grid> */}
              </Grid>
              <Button
                type="submit"
                fullWidth
                variant="contained"
                sx={{ mt: 3, mb: 2 }}
              >
                Analysis/Predict
              </Button>
              {/* <Grid container justifyContent="flex-end">
                <Grid item>
                  <Link href="#" variant="body2">
                    Already have an account? Sign in
                  </Link>
                </Grid>
              </Grid> */}
            </Box>
          </Box>
          <Copyright sx={{ mt: 5 }} />
        </Container>
      </ThemeProvider>
    );
  }
}

export default sym;
