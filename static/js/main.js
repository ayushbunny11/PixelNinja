$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('.imageRes').hide();



    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        readURL(this);
    });

    function showProcessedImage(imageData){
        var resImage= document.getElementById("result-image");
        resImage.setAttribute('src','data:image/jpeg;base64,'+imageData);
    }


    document.getElementById("btn-download").disabled = true;
    // Predict for Button Start
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        $('.loader').show();


        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('#btn-predict').show();
                document.getElementById("btn-predict").disabled = true;
                $('.imageRes').show();
                console.log('Success!');
                alert('Processing is done! Now you can download.')
                document.getElementById("btn-download").disabled = false;

                // Image Display function after Success of Image processing

                $.ajax({
                    url: '/processed-image',
                    method: 'GET',
                    success: function (response) {
                        // On success, call the showProcessedImage() function with the URL of the processed image
                        showProcessedImage(response.imageData);
                    }
                });
            },
        });
    });


    // Downlaod for Button Download
    $('#btn-download').click(function () {


        // Show loading animation
        $(this).hide();
        $('.loader').show();


        // Downlaoding the image
        $.ajax({
            type: 'GET',
            url: '/download',
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function () {
                // Get and display the result
                $('.loader').hide();
                $('#btn-download').show();
                document.getElementById("btn-download").disabled = true;

                alert('Check your downloads folder!')
                console.log('Success!');

            },
            error: function (jqXHR) {

                $('.loader').hide();
                $('#btn-download').show();
                document.getElementById("btn-download").disabled = true;
                alert('ERROR IN DOWNLOADING!!! Error Status Code: ' + jqXHR.status)

            },
        });
    });





});
